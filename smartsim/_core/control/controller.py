# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import itertools
import os.path as osp
import pathlib
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time
import typing as t
from os import environ

from smartredis import Client, ConfigOptions

from smartsim._core.types import JobIdType, StepName
from smartsim._core.utils.network import get_ip_from_host

from ..._core.launcher.step import Step
from ..._core.utils.helpers import (
    SignalInterceptionStack,
    unpack_colo_db_identifier,
    unpack_db_identifier,
)
from ..._core.utils.redis import (
    db_is_active,
    set_ml_model,
    set_script,
    shutdown_db_node,
)
from ...database import Orchestrator
from ...entity import Ensemble, EntitySequence, Model, SmartSimEntity
from ...error import (
    LauncherError,
    SmartSimError,
    SSDBIDConflictError,
    SSInternalError,
    SSUnsupportedError,
)
from ...log import get_logger
from ...servertype import CLUSTERED, STANDALONE
from ...status import TERMINAL_STATUSES, SmartSimStatus
from ..config import CONFIG
from ..launcher import LocalLauncher, LSFLauncher, PBSLauncher, SlurmLauncher
from ..launcher.launcher import Launcher
from ..utils import check_cluster_status, create_cluster, serialize
from .controller_utils import (
    SerializableLaunchedDBConfig as _SerializableLaunchedDBConfig,
)
from .controller_utils import (
    SerializableLaunchedDBStepInfo as _SerializableLaunchedDBStepInfo,
)
from .controller_utils import _AnonymousBatchJob, _look_up_launched_data
from .job import Job
from .jobmanager import JobManager
from .manifest import LaunchedManifest, LaunchedManifestBuilder, Manifest

if t.TYPE_CHECKING:
    from types import FrameType

    from ...entity import types as _entity_types
    from .. import types as _core_types
    from ..launcher.stepMapping import StepMap
    from ..utils.serialize import TStepLaunchMetaData
    from .job import JobEntity


logger = get_logger(__name__)

# job manager lock
JM_LOCK = threading.RLock()


class Controller:
    """The controller module provides an interface between the
    smartsim entities created in the experiment and the
    underlying workload manager or run framework.
    """

    # XXX: Ideally this map should not exist. A controller requires a launcher,
    #      so a launcher instance should just be passed in as part of
    #      construction...
    _LAUNCHER_MAP: t.Dict[str, t.Type[Launcher]] = {
        "slurm": SlurmLauncher,
        "pbs": PBSLauncher,
        "pals": PBSLauncher,
        "lsf": LSFLauncher,
        "local": LocalLauncher,
    }

    def __init__(self, launcher: str = "local") -> None:
        """Initialize a Controller

        :param launcher: the type of launcher being used
        :type launcher: str
        """
        launcher = launcher.lower()
        try:
            self._launcher = type(self)._LAUNCHER_MAP[launcher]()
        except KeyError:
            msg = f"Launcher type not supported: {launcher}"
            raise SSUnsupportedError(msg) from None
        jm_interval = (
            CONFIG.jm_interval if isinstance(self._launcher, LocalLauncher) else 2
        )
        self._job_manager = JobManager(JM_LOCK, poll_status_interval=jm_interval)
        self._telemetry_monitor: t.Optional[subprocess.Popen[bytes]] = None
        self._entity_to_job_monitor_id: t.Dict[
            "_entity_types.EntityName",
            # XXX: or should this be a
            #      `t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]`?
            #       Going with the name for now for ease of implementation
            "_core_types.MonitoredJobID",
        ] = {}

    def start(
        self,
        exp_name: str,
        exp_path: str,
        manifest: Manifest,
        block: bool = True,
        kill_on_interrupt: bool = True,
    ) -> None:
        """Start the passed SmartSim entities

        This function should not be called directly, but rather
        through the experiment interface.

        The controller will start the job-manager thread upon
        execution of all jobs.
        """
        self._job_manager.kill_on_interrupt = kill_on_interrupt

        # register custom signal handler for ^C (SIGINT)
        SignalInterceptionStack.get(signal.SIGINT).push_unique(
            self._job_manager.signal_interrupt
        )
        launched = self._launch(exp_name, exp_path, manifest)

        # start the job manager thread if not already started
        if not self._job_manager.actively_monitoring:
            self._job_manager.start()

        serialize.save_launch_manifest(
            launched.map(_look_up_launched_data(self._launcher))
        )

        # launch a telemetry monitor to track job progress
        if CONFIG.telemetry_enabled:
            self._start_telemetry_monitor(exp_path)

        # block until all non-database jobs are complete
        if block:
            # poll handles its own keyboard interrupt as
            # it may be called separately
            self.poll(5, True, kill_on_interrupt=kill_on_interrupt)

    @property
    def orchestrator_active(self) -> bool:
        return len(self._job_manager.ongoing_db_jobs) > 0

    def poll(
        self, interval: int, verbose: bool, kill_on_interrupt: bool = True
    ) -> None:
        """Poll running jobs and receive logging output of job status

        :param interval: number of seconds to wait before polling again
        :type interval: int
        :param verbose: set verbosity
        :type verbose: bool
        :param kill_on_interrupt: flag for killing jobs when SIGINT is received
        :type kill_on_interrupt: bool, optional
        """
        self._job_manager.kill_on_interrupt = kill_on_interrupt
        while jobs := self._job_manager.ongoing_non_db_jobs:
            time.sleep(interval)
            if verbose:
                for job in jobs:
                    logger.info(job)

    def finished(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> bool:
        """Return a boolean indicating whether a job has finished or not

        :param entity: object launched by SmartSim.
        :type entity: Entity | EntitySequence
        :returns: bool
        :raises ValueError: if entity has not been launched yet
        """
        try:
            if isinstance(entity, Orchestrator):
                raise TypeError("Finished() does not support Orchestrator instances")
            if isinstance(entity, EntitySequence):
                return all(self.finished(ent) for ent in entity.entities)
            if not isinstance(entity, SmartSimEntity):
                raise TypeError(
                    f"Argument was of type {type(entity)} not derived "
                    "from SmartSimEntity or EntitySequence"
                )

            return self._job_manager.is_finished(
                self._entity_to_job_monitor_id[entity.name]
            )
        except KeyError:
            raise ValueError(
                f"Entity {entity.name} has not been launched in this experiment"
            ) from None

    def _stop(
        self,
        entities: t.Sequence[t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]],
    ) -> None:
        ids = tuple(self._entity_to_job_monitor_id[entity.name] for entity in entities)
        self._job_manager.stop_jobs(ids)

    def stop_entity(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> None:
        """Stop an instance of an entity

        This function will also update the status of the job in
        the jobmanager so that the job appears as "cancelled".

        :param entity: entity to be stopped
        :type entity: Entity | EntitySequence
        """
        self._stop((entity,))

    def stop_db(self, db: Orchestrator) -> None:
        """Stop an orchestrator
        :param db: orchestrator to be stopped
        :type db: Orchestrator
        """
        if db.batch:
            self._stop((db,))
        else:
            self._stop(db.entities)
        db.reset_hosts()

    def stop_entity_list(self, entity_list: EntitySequence[SmartSimEntity]) -> None:
        """Stop an instance of an entity list

        :param entity_list: entity list to be stopped
        :type entity_list: EntitySequence
        """

        if entity_list.batch:
            self._stop((entity_list,))
        else:
            self._stop(entity_list.entities)

    def get_jobs(self) -> t.Dict["_core_types.MonitoredJobID", Job]:
        """Return a dictionary of completed job data

        :returns: dict[str, Job]
        """
        return self._job_manager.completed

    def get_entity_status(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> SmartSimStatus:
        """Get the status of an entity

        :param entity: entity to get status of
        :type entity: SmartSimEntity | EntitySequence
        :raises TypeError: if not SmartSimEntity | EntitySequence
        :return: status of entity
        :rtype: SmartSimStatus
        """
        if not isinstance(entity, (SmartSimEntity, EntitySequence)):
            raise TypeError(
                "Argument must be of type SmartSimEntity or EntitySequence, "
                f"not {type(entity)}"
            )
        id_ = self._entity_to_job_monitor_id[entity.name]
        return self._job_manager.get_status(id_)

    def get_entity_list_status(
        self, entity_list: EntitySequence[SmartSimEntity]
    ) -> t.List[SmartSimStatus]:
        """Get the statuses of an entity list

        :param entity_list: entity list containing entities to
                            get statuses of
        :type entity_list: EntitySequence
        :raises TypeError: if not EntitySequence
        :return: list of SmartSimStatus statuses
        :rtype: list
        """
        if not isinstance(entity_list, EntitySequence):
            raise TypeError(
                f"Argument was of type {type(entity_list)} not EntitySequence"
            )
        if entity_list.batch:
            return [self.get_entity_status(entity_list)]
        statuses = []
        for entity in entity_list.entities:
            statuses.append(self.get_entity_status(entity))
        return statuses

    @staticmethod
    def symlink_output_files(
        job_step: Step, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> None:
        """Create symlinks for entity output files that point to the output files
        under the .smartsim directory

        :param job_step: Job step instance
        :type job_step: Step
        :param entity: Entity instance
        :type entity: SmartSimEntity | EntitySequence[SmartSimEntity]
        """
        historical_out, historical_err = map(pathlib.Path, job_step.get_output_files())
        entity_out = pathlib.Path(entity.path) / f"{entity.name}.out"
        entity_err = pathlib.Path(entity.path) / f"{entity.name}.err"

        # check if there is already a link to a previous run
        if entity_out.is_symlink() or entity_err.is_symlink():
            entity_out.unlink()
            entity_err.unlink()

        historical_err.touch()
        historical_out.touch()

        if historical_err.exists() and historical_out.exists():
            entity_out.symlink_to(historical_out)
            entity_err.symlink_to(historical_err)
        else:
            raise FileNotFoundError(
                f"Output files for {entity.name} could not be found. "
                "Symlinking files failed."
            )

    def _launch(
        self, exp_name: str, exp_path: str, manifest: Manifest
    ) -> LaunchedManifest[t.Tuple[StepName, Step]]:
        """Main launching function of the controller

        Orchestrators are always launched first so that the
        address of the database can be given to following entities

        :param exp_name: The name of the launching experiment
        :type exp_name: str
        :param exp_path: path to location of ``Experiment`` directory if generated
        :type exp_path: str
        :param manifest: Manifest of deployables to launch
        :type manifest: Manifest
        """

        manifest_builder = LaunchedManifestBuilder[t.Tuple[StepName, Step]](
            exp_name=exp_name,
            exp_path=exp_path,
            launcher_name=str(self._launcher),
        )
        # Loop over deployables to launch and launch multiple orchestrators
        for orchestrator in manifest.dbs:
            for key in self._job_manager.get_db_host_addresses():
                _, db_id = unpack_db_identifier(key, "_")
                if orchestrator.db_identifier == db_id:
                    raise SSDBIDConflictError(
                        f"Database identifier {orchestrator.db_identifier}"
                        " has already been used. Pass in a unique"
                        " name for db_identifier"
                    )

            if orchestrator.num_shards > 1 and isinstance(
                self._launcher, LocalLauncher
            ):
                raise SmartSimError(
                    "Local launcher does not support multi-host orchestrators"
                )
            self._launch_orchestrator(orchestrator, manifest_builder)

        if self.orchestrator_active:
            self._set_dbobjects(manifest)

        # create all steps prior to launch
        steps: t.List[
            t.Tuple[Step, t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]]
        ] = []

        symlink_substeps: t.List[
            t.Tuple[Step, t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]]
        ] = []

        for elist in manifest.ensembles:
            ens_telem_dir = manifest_builder.run_telemetry_subdirectory / "ensemble"
            if elist.batch:
                batch_step, substeps = self._create_batch_job_step(elist, ens_telem_dir)
                manifest_builder.add_ensemble(
                    elist, [(batch_step.name, step) for step in substeps]
                )

                # symlink substeps to maintain directory structure
                for substep, substep_entity in zip(substeps, elist.models):
                    symlink_substeps.append((substep, substep_entity))

                steps.append((batch_step, elist))
            else:
                # if ensemble is to be run as separate job steps, aka not in a batch
                job_steps = [
                    (self._create_job_step(e, ens_telem_dir / elist.name), e)
                    for e in elist.entities
                ]
                manifest_builder.add_ensemble(
                    elist, [(step.name, step) for step, _ in job_steps]
                )
                steps.extend(job_steps)
        # models themselves cannot be batch steps. If batch settings are
        # attached, wrap them in an anonymous batch job step
        for model in manifest.models:
            model_telem_dir = manifest_builder.run_telemetry_subdirectory / "model"
            if model.batch_settings:
                anon_entity_list = _AnonymousBatchJob(model)
                batch_step, substeps = self._create_batch_job_step(
                    anon_entity_list, model_telem_dir
                )
                manifest_builder.add_model(model, (batch_step.name, batch_step))

                symlink_substeps.append((substeps[0], model))
                steps.append((batch_step, model))
            else:
                job_step = self._create_job_step(model, model_telem_dir)
                manifest_builder.add_model(model, (job_step.name, job_step))
                steps.append((job_step, model))

        # launch and symlink steps
        for step, entity in steps:
            self._launch_step(step, entity)
            self.symlink_output_files(step, entity)

        # symlink substeps to maintain directory structure
        for substep, entity in symlink_substeps:
            self.symlink_output_files(substep, entity)

        return manifest_builder.finalize()

    def _launch_orchestrator(
        self,
        orchestrator: Orchestrator,
        manifest_builder: LaunchedManifestBuilder[t.Tuple[StepName, Step]],
    ) -> None:
        """Launch an Orchestrator instance

        This function will launch the Orchestrator instance and
        if on WLM, find the nodes where it was launched and
        set them in the JobManager

        :param orchestrator: orchestrator to launch
        :type orchestrator: Orchestrator
        :param manifest_builder: An `LaunchedManifestBuilder` to record the
                                 names and `Step`s of the launched orchestrator
        :type manifest_builder: LaunchedManifestBuilder[tuple[str, Step]]
        """
        orchestrator.remove_stale_files()
        orc_telem_dir = manifest_builder.run_telemetry_subdirectory / "database"

        # if the orchestrator was launched as a batch workload
        if orchestrator.batch:
            orc_batch_step, substeps = self._create_batch_job_step(
                orchestrator, orc_telem_dir
            )
            manifest_builder.add_database(
                orchestrator, [(orc_batch_step.name, step) for step in substeps]
            )

            self._launch_step(orc_batch_step, orchestrator)
            self.symlink_output_files(orc_batch_step, orchestrator)

            # symlink substeps to maintain directory structure
            for substep, substep_entity in zip(substeps, orchestrator.entities):
                self.symlink_output_files(substep, substep_entity)

            launched_steps: t.Tuple[Step, ...] = (orc_batch_step,)

        # if orchestrator was run on existing allocation, locally, or in allocation
        else:
            db_steps = [
                (self._create_job_step(db, orc_telem_dir / orchestrator.name), db)
                for db in orchestrator.entities
            ]
            manifest_builder.add_database(
                orchestrator, [(step.name, step) for step, _ in db_steps]
            )
            for db_step in db_steps:
                self._launch_step(*db_step)
                self.symlink_output_files(*db_step)
            launched_steps = tuple(step for step, _ in db_steps)

        # wait for orchestrator to spin up
        self._orchestrator_launch_wait(orchestrator)

        # set the jobs in the job manager to provide SSDB variable to entities
        # if _host isnt set within each

        # create the database cluster
        if orchestrator.num_shards > 2:
            num_trials = 5
            cluster_created = False
            while not cluster_created:
                try:
                    create_cluster(orchestrator.hosts, orchestrator.ports)
                    check_cluster_status(orchestrator.hosts, orchestrator.ports)
                    num_shards = orchestrator.num_shards
                    logger.info(f"Database cluster created with {num_shards} shards")
                    cluster_created = True
                except SSInternalError:
                    if num_trials > 0:
                        logger.debug(
                            "Cluster creation failed, attempting again in five seconds."
                        )
                        num_trials -= 1
                        time.sleep(5)
                    else:
                        # surface SSInternalError as we have no way to recover
                        raise
        self._save_orchestrator(
            orchestrator,
            (
                (
                    self._job_manager[self._entity_to_job_monitor_id[step.entity_name]],
                    step,
                    self._launcher.step_mapping[step.name],
                )
                for step in launched_steps
            ),
        )
        logger.debug(f"Orchestrator launched on nodes: {orchestrator.hosts}")

    def _launch_step(
        self,
        job_step: Step,
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]],
    ) -> None:
        """Use the launcher to launch a job step

        :param job_step: a job step instance
        :type job_step: Step
        :param entity: entity instance
        :type entity: SmartSimEntity
        :raises SmartSimError: if launch fails
        """
        # attempt to retrieve entity name in JobManager.completed
        try:
            id_ = self._entity_to_job_monitor_id[entity.name]
        except KeyError:
            completed_job = None
        else:
            completed_job = self._job_manager.find_completed_job(id_)

        # If completed job DNE and is the entity name is not tracked by the
        # ``JobManager``, launch the job
        if completed_job is None:
            try:
                job_id = self._launcher.run(job_step)
            except LauncherError as e:
                msg = textwrap.dedent(f"""\
                    An error occurred when launching {entity.name}
                    Check error and output files for details.
                    """) + str(entity)
                logger.error(msg)
                raise SmartSimError(f"Job step {entity.name} failed to launch") from e
            job = Job.from_launched_step(job_id, self._launcher, job_step, entity)
            #                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # XXX: All of this data aside of the entity is theoretically known
            #      by the launcher at the moment it "runs" a step, and the only
            #      thing you can do with a step is "run" it, and, to create a
            #      step, a launcher needs to take in an entity! Why does
            #      `Launcher.run` not just take in an entity, and return `Job`?
            #      Do we use `Step`s literally anywhere else??
            self._entity_to_job_monitor_id[entity.name] = self._job_manager.add_job(job)

        # if the completed job does exist and the entity passed in is the same
        # that has ran and completed, relaunch the entity.
        elif completed_job is not None and completed_job.entity is entity:
            try:
                job_id = self._launcher.run(job_step)
            except LauncherError as e:
                msg = textwrap.dedent(f"""\
                    An error occurred when launching {entity.name}
                    Check error and output files for details.
                    """) + str(entity)
                logger.error(msg)
                raise SmartSimError(f"Job step {entity.name} failed to launch") from e

            self._job_manager.restart_job(
                job_step.name,
                # ^^^^^^^^^^^
                job_id,
                self._entity_to_job_monitor_id[entity.name],
                not job_step.managed,
                # ^^^^^^^^^^^^^^^^^^
            )
            # XXX: These fields can be retrieved off of the existing job, and
            #      are (or at least we assume are) immutable. Why do we need to
            #      respecify them here??

        # the entity is using a duplicate name of an existing entity in
        # the experiment, throw an error
        else:
            raise SSUnsupportedError("SmartSim entities cannot have duplicate names.")

    def _create_batch_job_step(
        self,
        entity_list: t.Union[Orchestrator, Ensemble, _AnonymousBatchJob],
        telemetry_dir: pathlib.Path,
    ) -> t.Tuple[Step, t.List[Step]]:
        """Use launcher to create batch job step

        :param entity_list: EntityList to launch as batch
        :type entity_list: EntityList
        :param telemetry_dir: Path to a directory in which the batch job step
                              may write telemetry events
        :type telemetry_dir: pathlib.Path
        :return: batch job step instance and a list of run steps to be
                 executed within the batch job
        :rtype: tuple[Step, list[Step]]
        """
        telemetry_dir = telemetry_dir / entity_list.name
        batch_step = self._launcher.create_batch_step_from(entity_list)
        batch_step.meta["entity_type"] = str(type(entity_list).__name__).lower()
        batch_step.meta["status_dir"] = str(telemetry_dir)

        substeps = []
        for entity in entity_list.entities:
            # tells step creation not to look for an allocation
            entity.run_settings.in_batch = True
            step = self._create_job_step(entity, telemetry_dir)
            substeps.append(step)
            batch_step.add_to_batch(step)
        return batch_step, substeps

    def _create_job_step(
        self, entity: SmartSimEntity, telemetry_dir: pathlib.Path
    ) -> Step:
        """Create job steps for all entities with the launcher

        :param entity: an entity to create a step for
        :type entity: SmartSimEntity
        :param telemetry_dir: Path to a directory in which the job step
                               may write telemetry events
        :type telemetry_dir: pathlib.Path
        :return: the job step
        :rtype: Step
        """
        # get SSDB, SSIN, SSOUT and add to entity run settings
        if isinstance(entity, Model):
            self._prep_entity_client_env(entity)

        step = self._launcher.create_step_from(entity)
        step.meta["entity_type"] = str(type(entity).__name__).lower()
        step.meta["status_dir"] = str(telemetry_dir / entity.name)
        return step

    def _prep_entity_client_env(self, entity: Model) -> None:
        """Retrieve all connections registered to this entity

        :param entity: The entity to retrieve connections from
        :type entity:  Model
        """

        client_env: t.Dict[str, t.Union[str, int, float, bool]] = {}
        address_dict = self._job_manager.get_db_host_addresses()

        for db_id, addresses in address_dict.items():
            db_name, _ = unpack_db_identifier(db_id, "_")
            if addresses:
                # Cap max length of SSDB
                client_env[f"SSDB{db_name}"] = ",".join(addresses[:128])

                # Retrieve num_shards to append to client env
                client_env[f"SR_DB_TYPE{db_name}"] = (
                    CLUSTERED if len(addresses) > 1 else STANDALONE
                )

        if entity.incoming_entities:
            client_env["SSKEYIN"] = ",".join(
                [in_entity.name for in_entity in entity.incoming_entities]
            )
        if entity.query_key_prefixing():
            client_env["SSKEYOUT"] = entity.name

        # Set address to local if it's a colocated model
        if entity.colocated and entity.run_settings.colocated_db_settings is not None:
            db_name_colo = entity.run_settings.colocated_db_settings["db_identifier"]
            assert isinstance(db_name_colo, str)
            for key in address_dict:
                _, db_id = unpack_db_identifier(key, "_")
                if db_name_colo == db_id:
                    raise SSDBIDConflictError(
                        f"Database identifier {db_name_colo}"
                        " has already been used. Pass in a unique"
                        " name for db_identifier"
                    )

            db_name_colo = unpack_colo_db_identifier(db_name_colo)
            if colo_cfg := entity.run_settings.colocated_db_settings:
                port = colo_cfg.get("port", None)
                socket = colo_cfg.get("unix_socket", None)
                if socket and port:
                    raise SSInternalError(
                        "Co-located was configured for both TCP/IP and UDS"
                    )
                if port:
                    client_env[f"SSDB{db_name_colo}"] = f"127.0.0.1:{str(port)}"
                elif socket:
                    client_env[f"SSDB{db_name_colo}"] = f"unix://{socket}"
                else:
                    raise SSInternalError(
                        "Colocated database was not configured for either TCP or UDS"
                    )
                client_env[f"SR_DB_TYPE{db_name_colo}"] = STANDALONE

        entity.run_settings.update_env(client_env)

    def _save_orchestrator(
        self,
        orchestrator: Orchestrator,
        launched_job_steps: t.Iterable[t.Tuple[Job, Step, "StepMap"]],
    ) -> None:
        """Save the orchestrator object via pickle

        This function saves the orchestrator information to a pickle
        file that can be imported by subsequent experiments to reconnect
        to the orchestrator.

        :param orchestrator: Orchestrator configuration to be saved
        :type orchestrator: Orchestrator
        """

        dat_file = osp.join(orchestrator.path, "smartsim_db.dat")
        orc_data = _SerializableLaunchedDBConfig(
            orchestrator,
            tuple(
                _SerializableLaunchedDBStepInfo(job.jid, step, map_, job.entity)
                for job, step, map_ in launched_job_steps
            ),
        )
        with open(dat_file, "wb") as pickle_file:
            pickle.dump(orc_data, pickle_file)

    def _orchestrator_launch_wait(self, orchestrator: Orchestrator) -> None:
        """Wait for the orchestrator instances to run

        In the case where the orchestrator is launched as a batch
        through a WLM, we wait for the orchestrator to exit the
        queue before proceeding so new launched entities can
        be launched with SSDB address

        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        :raises SmartSimError: if launch fails or manually stopped by user
        """
        if orchestrator.batch:
            logger.info("Orchestrator launched as a batch")
            logger.info("While queued, SmartSim will wait for Orchestrator to run")
            logger.info("CTRL+C interrupt to abort and cancel launch")

        ready = False
        while not ready:
            try:
                time.sleep(CONFIG.jm_interval)
                # manually trigger job update if JM not running
                if not self._job_manager.actively_monitoring:
                    self._job_manager.update_statuses()

                # _job_manager.get_status acquires JM lock for main thread, no need for locking
                statuses = self.get_entity_list_status(orchestrator)
                if all(stat == SmartSimStatus.STATUS_RUNNING for stat in statuses):
                    ready = True
                    # TODO remove in favor of by node status check
                    time.sleep(CONFIG.jm_interval)
                elif any(stat in TERMINAL_STATUSES for stat in statuses):
                    self.stop_db(orchestrator)
                    msg = "Orchestrator failed during startup"
                    msg += f" See {orchestrator.path} for details"
                    raise SmartSimError(msg)
                else:
                    logger.debug("Waiting for orchestrator instances to spin up...")
            except KeyboardInterrupt:
                logger.info("Orchestrator launch cancelled - requesting to stop")
                self.stop_db(orchestrator)

                # re-raise keyboard interrupt so the job manager will display
                # any running and un-killed jobs as this method is only called
                # during launch and we handle all keyboard interrupts during
                # launch explicitly
                raise

    def reload_saved_db(self, checkpoint_file: str) -> Orchestrator:
        try:
            with open(checkpoint_file, "rb") as pickle_file:
                db_config = pickle.load(pickle_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The SmartSim database config file {checkpoint_file} "
                "cannot be found."
            ) from None
        except (OSError, IOError) as e:
            msg = "Database checkpoint corrupted"
            raise SmartSimError(msg) from e

        if not isinstance(db_config, _SerializableLaunchedDBConfig):
            raise SmartSimError(
                "The SmartSim database checkpoint is incomplete or corrupted."
            )

        # TODO check that each db_object is running

        job_steps = (
            (
                Job.from_launched_step(
                    info.job_id, self._launcher, info.step, info.entity
                ),
                info.step_map,
            )
            for info in db_config.launched_step_info
        )

        # XXX: Currently this strategy will lose job history and status (at
        #      least until the JM has a chance to reset it). Is that going
        #      to be a problem?
        try:
            for db_job, step in job_steps:
                self._entity_to_job_monitor_id[db_job.ename] = (
                    self._job_manager.add_job(db_job)
                )
                self._launcher.step_mapping[db_job.name] = step
                if step.task_id:
                    self._launcher.task_manager.add_existing(int(step.task_id))
        except LauncherError as e:
            raise SmartSimError("Failed to reconnect orchestrator") from e

        # start job manager if not already started
        if not self._job_manager.actively_monitoring:
            self._job_manager.start()

        return db_config.database

    def _set_dbobjects(self, manifest: Manifest) -> None:
        if not manifest.has_db_objects:
            return

        address_dict = self._job_manager.get_db_host_addresses()
        for (
            db_id,
            db_addresses,
        ) in address_dict.items():
            db_name, name = unpack_db_identifier(db_id, "_")

            hosts = list({address.split(":")[0] for address in db_addresses})
            ports = list({int(address.split(":")[-1]) for address in db_addresses})

            if not db_is_active(hosts=hosts, ports=ports, num_shards=len(db_addresses)):
                raise SSInternalError("Cannot set DB Objects, DB is not running")

            environ[f"SSDB{db_name}"] = db_addresses[0]

            environ[f"SR_DB_TYPE{db_name}"] = (
                CLUSTERED if len(db_addresses) > 1 else STANDALONE
            )

            options = ConfigOptions.create_from_environment(name)
            client = Client(options, logger_name="SmartSim")

            for model in manifest.models:
                if not model.colocated:
                    for db_model in model.db_models:
                        set_ml_model(db_model, client)
                    for db_script in model.db_scripts:
                        set_script(db_script, client)

            for ensemble in manifest.ensembles:
                for db_model in ensemble.db_models:
                    set_ml_model(db_model, client)
                for db_script in ensemble.db_scripts:
                    set_script(db_script, client)
                for entity in ensemble.models:
                    if not entity.colocated:
                        # Set models which could belong only
                        # to the entities and not to the ensemble
                        # but avoid duplicates
                        for db_model in entity.db_models:
                            if db_model not in ensemble.db_models:
                                set_ml_model(db_model, client)
                        for db_script in entity.db_scripts:
                            if db_script not in ensemble.db_scripts:
                                set_script(db_script, client)

    def _start_telemetry_monitor(self, exp_dir: str) -> None:
        """Spawns a telemetry monitor process to keep track of the life times
        of the processes launched through this controller.

        :param exp_dir: An experiment directory
        :type exp_dir: str
        """
        if (
            self._telemetry_monitor is None
            or self._telemetry_monitor.returncode is not None
        ):

            logger.debug("Starting telemetry monitor process")
            cmd = [
                sys.executable,
                "-m",
                "smartsim._core.entrypoints.telemetrymonitor",
                "-exp_dir",
                exp_dir,
                "-frequency",
                str(CONFIG.telemetry_frequency),
                "-cooldown",
                str(CONFIG.telemetry_cooldown),
            ]
            # pylint: disable-next=consider-using-with
            self._telemetry_monitor = subprocess.Popen(
                cmd,
                stderr=sys.stderr,
                stdout=sys.stdout,
                cwd=str(pathlib.Path(__file__).parent.parent.parent),
                shell=False,
            )
