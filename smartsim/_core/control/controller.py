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
import threading
import time
import typing as t
from os import environ

from smartredis import Client, ConfigOptions

from smartsim._core.utils.network import get_ip_from_host

from ..._core.launcher.step import Step
from ..._core.utils.helpers import unpack_colo_db_identifier, unpack_db_identifier
from ..._core.utils.redis import (
    db_is_active,
    set_ml_model,
    set_script,
    shutdown_db_node,
)
from ...database import Orchestrator
from ...entity import Ensemble, EntityList, EntitySequence, Model, SmartSimEntity
from ...error import (
    LauncherError,
    SmartSimError,
    SSDBIDConflictError,
    SSInternalError,
    SSUnsupportedError,
)
from ...log import get_logger
from ...servertype import CLUSTERED, STANDALONE
from ...status import STATUS_CANCELLED, STATUS_RUNNING, TERMINAL_STATUSES
from ..config import CONFIG
from ..launcher import LocalLauncher, LSFLauncher, PBSLauncher, SlurmLauncher
from ..launcher.launcher import Launcher
from ..utils import check_cluster_status, create_cluster, serialize
from .job import Job
from .jobmanager import JobManager
from .manifest import LaunchedManifest, LaunchedManifestBuilder, Manifest

if t.TYPE_CHECKING:
    from ..utils.serialize import TStepLaunchMetaData


logger = get_logger(__name__)

# job manager lock
JM_LOCK = threading.RLock()


class Controller:
    """The controller module provides an interface between the
    smartsim entities created in the experiment and the
    underlying workload manager or run framework.
    """

    def __init__(self, launcher: str = "local") -> None:
        """Initialize a Controller

        :param launcher: the type of launcher being used
        :type launcher: str
        """
        self._jobs = JobManager(JM_LOCK)
        self.init_launcher(launcher)
        self._telemetry_monitor: t.Optional[subprocess.Popen[bytes]] = None

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
        self._jobs.kill_on_interrupt = kill_on_interrupt
        # register custom signal handler for ^C (SIGINT)
        signal.signal(signal.SIGINT, self._jobs.signal_interrupt)
        launched = self._launch(exp_name, exp_path, manifest)

        # start the job manager thread if not already started
        if not self._jobs.actively_monitoring:
            self._jobs.start()

        serialize.save_launch_manifest(
            launched.map(_look_up_launched_data(self._launcher))
        )

        # launch a telemetry monitor to track job progress
        if CONFIG.telemetry_enabled:
            self._start_telemetry_monitor(exp_path)

        # block until all non-database jobs are complete
        if block:
            # poll handles its own keyboard interrupt as
            # it may be called seperately
            self.poll(5, True, kill_on_interrupt=kill_on_interrupt)

    @property
    def orchestrator_active(self) -> bool:
        with JM_LOCK:
            if len(self._jobs.db_jobs) > 0:
                return True
            return False

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
        self._jobs.kill_on_interrupt = kill_on_interrupt
        to_monitor = self._jobs.jobs
        while len(to_monitor) > 0:
            time.sleep(interval)

            # acquire lock to avoid "dictionary changed during iteration" error
            # without having to copy dictionary each time.
            if verbose:
                with JM_LOCK:
                    for job in to_monitor.values():
                        logger.info(job)

    def finished(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> bool:
        """Return a boolean indicating wether a job has finished or not

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

            return self._jobs.is_finished(entity)
        except KeyError:
            raise ValueError(
                f"Entity {entity.name} has not been launched in this experiment"
            ) from None

    def stop_entity(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> None:
        """Stop an instance of an entity

        This function will also update the status of the job in
        the jobmanager so that the job appears as "cancelled".

        :param entity: entity to be stopped
        :type entity: Entity | EntitySequence
        """
        with JM_LOCK:
            job = self._jobs[entity.name]
            if job.status not in TERMINAL_STATUSES:
                logger.info(
                    " ".join(
                        ("Stopping model", entity.name, "with job name", str(job.name))
                    )
                )
                status = self._launcher.stop(job.name)

                job.set_status(
                    status.status,
                    status.launcher_status,
                    status.returncode,
                    error=status.error,
                    output=status.output,
                )
                self._jobs.move_to_completed(job)

    def stop_db(self, db: Orchestrator) -> None:
        """Stop an orchestrator
        :param db: orchestrator to be stopped
        :type db: Orchestrator
        """
        if db.batch:
            self.stop_entity(db)
        else:
            with JM_LOCK:
                for node in db.entities:
                    for host_ip, port in itertools.product(
                        (get_ip_from_host(host) for host in node.hosts), db.ports
                    ):
                        retcode, _, _ = shutdown_db_node(host_ip, port)
                        # Sometimes the DB will not shutdown (unless we force NOSAVE)
                        if retcode != 0:
                            self.stop_entity(node)
                            continue

                        job = self._jobs[node.name]
                        job.set_status(STATUS_CANCELLED, "", 0, output=None, error=None)
                        self._jobs.move_to_completed(job)

        db.reset_hosts()

    def stop_entity_list(self, entity_list: EntitySequence[SmartSimEntity]) -> None:
        """Stop an instance of an entity list

        :param entity_list: entity list to be stopped
        :type entity_list: EntitySequence
        """

        if entity_list.batch:
            self.stop_entity(entity_list)
        else:
            for entity in entity_list.entities:
                self.stop_entity(entity)

    def get_jobs(self) -> t.Dict[str, Job]:
        """Return a dictionary of completed job data

        :returns: dict[str, Job]
        """
        with JM_LOCK:
            return self._jobs.completed

    def get_entity_status(
        self, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> str:
        """Get the status of an entity

        :param entity: entity to get status of
        :type entity: SmartSimEntity | EntitySequence
        :raises TypeError: if not SmartSimEntity | EntitySequence
        :return: status of entity
        :rtype: str
        """
        if not isinstance(entity, (SmartSimEntity, EntitySequence)):
            raise TypeError(
                "Argument must be of type SmartSimEntity or EntitySequence, "
                f"not {type(entity)}"
            )
        return self._jobs.get_status(entity)

    def get_entity_list_status(
        self, entity_list: EntitySequence[SmartSimEntity]
    ) -> t.List[str]:
        """Get the statuses of an entity list

        :param entity_list: entity list containing entities to
                            get statuses of
        :type entity_list: EntitySequence
        :raises TypeError: if not EntitySequence
        :return: list of str statuses
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

    def init_launcher(self, launcher: str) -> None:
        """Initialize the controller with a specific type of launcher.
        SmartSim currently supports slurm, pbs(pro), lsf,
        and local launching

        :param launcher: which launcher to initialize
        :type launcher: str
        :raises SSUnsupportedError: if a string is passed that is not
                                    a supported launcher
        :raises TypeError: if no launcher argument is provided.
        """
        launcher_map: t.Dict[str, t.Type[Launcher]] = {
            "slurm": SlurmLauncher,
            "pbs": PBSLauncher,
            "pals": PBSLauncher,
            "lsf": LSFLauncher,
            "local": LocalLauncher,
        }

        if launcher is not None:
            launcher = launcher.lower()
            if launcher in launcher_map:
                # create new instance of the launcher
                self._launcher = launcher_map[launcher]()
                self._jobs.set_launcher(self._launcher)
            else:
                raise SSUnsupportedError("Launcher type not supported: " + launcher)
        else:
            raise TypeError("Must provide a 'launcher' argument")

    def _launch(
        self, exp_name: str, exp_path: str, manifest: Manifest
    ) -> LaunchedManifest[t.Tuple[str, Step]]:
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

        manifest_builder = LaunchedManifestBuilder[t.Tuple[str, Step]](
            exp_name=exp_name, exp_path=exp_path, launcher_name=str(self._launcher)
        )
        # Loop over deployables to launch and launch multiple orchestrators
        for orchestrator in manifest.dbs:
            for key in self._jobs.get_db_host_addresses():
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
        for elist in manifest.ensembles:
            ens_telem_dir = manifest_builder.run_telemetry_subdirectory / "ensemble"
            if elist.batch:
                batch_step, substeps = self._create_batch_job_step(elist, ens_telem_dir)
                manifest_builder.add_ensemble(
                    elist, [(batch_step.name, step) for step in substeps]
                )
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
                batch_step, _ = self._create_batch_job_step(
                    anon_entity_list, model_telem_dir
                )
                manifest_builder.add_model(model, (batch_step.name, batch_step))
                steps.append((batch_step, model))
            else:
                job_step = self._create_job_step(model, model_telem_dir)
                manifest_builder.add_model(model, (job_step.name, job_step))
                steps.append((job_step, model))

        # launch steps
        for step, entity in steps:
            self._launch_step(step, entity)

        return manifest_builder.finalize()

    def _launch_orchestrator(
        self,
        orchestrator: Orchestrator,
        manifest_builder: LaunchedManifestBuilder[t.Tuple[str, Step]],
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

        # wait for orchestrator to spin up
        self._orchestrator_launch_wait(orchestrator)

        # set the jobs in the job manager to provide SSDB variable to entities
        # if _host isnt set within each
        self._jobs.set_db_hosts(orchestrator)

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
        self._save_orchestrator(orchestrator)
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
        try:
            job_id = self._launcher.run(job_step)
        except LauncherError as e:
            msg = f"An error occurred when launching {entity.name} \n"
            msg += "Check error and output files for details.\n"
            msg += f"{entity}"
            logger.error(msg)
            raise SmartSimError(f"Job step {entity.name} failed to launch") from e

        # a job step is a task if it is not managed by a workload manager (i.e. Slurm)
        # but is rather started, monitored, and exited through the Popen interface
        # in the taskmanager
        is_task = not job_step.managed

        if self._jobs.query_restart(entity.name):
            logger.debug(f"Restarting {entity.name}")
            self._jobs.restart_job(job_step.name, job_id, entity.name, is_task)
        else:
            logger.debug(f"Launching {entity.name}")
            self._jobs.add_job(job_step.name, job_id, entity, is_task)

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
        if not entity_list.batch_settings:
            raise ValueError(
                "EntityList must have batch settings to be launched as batch"
            )

        telemetry_dir = telemetry_dir / entity_list.name
        batch_step = self._launcher.create_step(
            entity_list.name, entity_list.path, entity_list.batch_settings
        )
        batch_step.meta["entity_type"] = str(type(entity_list).__name__).lower()
        batch_step.meta["status_dir"] = str(telemetry_dir / entity_list.name)

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

        step = self._launcher.create_step(entity.name, entity.path, entity.run_settings)

        step.meta["entity_type"] = str(type(entity).__name__).lower()
        step.meta["status_dir"] = str(telemetry_dir / entity.name)

        return step

    def _prep_entity_client_env(self, entity: Model) -> None:
        """Retrieve all connections registered to this entity

        :param entity: The entity to retrieve connections from
        :type entity:  Model
        """

        client_env: t.Dict[str, t.Union[str, int, float, bool]] = {}
        address_dict = self._jobs.get_db_host_addresses()

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

    def _save_orchestrator(self, orchestrator: Orchestrator) -> None:
        """Save the orchestrator object via pickle

        This function saves the orchestrator information to a pickle
        file that can be imported by subsequent experiments to reconnect
        to the orchestrator.

        :param orchestrator: Orchestrator configuration to be saved
        :type orchestrator: Orchestrator
        """

        dat_file = "/".join((orchestrator.path, "smartsim_db.dat"))
        db_jobs = self._jobs.db_jobs
        orc_data = {"db": orchestrator, "db_jobs": db_jobs}
        steps = []
        for db_job in db_jobs.values():
            steps.append(self._launcher.step_mapping[db_job.name])
        orc_data["steps"] = steps
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
                if not self._jobs.actively_monitoring:
                    self._jobs.check_jobs()

                # _jobs.get_status acquires JM lock for main thread, no need for locking
                statuses = self.get_entity_list_status(orchestrator)
                if all(stat == STATUS_RUNNING for stat in statuses):
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
        with JM_LOCK:
            if self.orchestrator_active:
                raise SmartSimError("Orchestrator exists and is active")

            if not osp.exists(checkpoint_file):
                raise FileNotFoundError(
                    f"The SmartSim database config file {checkpoint_file} "
                    "cannot be found."
                )

            try:
                with open(checkpoint_file, "rb") as pickle_file:
                    db_config = pickle.load(pickle_file)
            except (OSError, IOError) as e:
                msg = "Database checkpoint corrupted"
                raise SmartSimError(msg) from e

            err_message = (
                "The SmartSim database checkpoint is incomplete or corrupted. "
            )
            if not "db" in db_config:
                raise SmartSimError(
                    err_message + "Could not find the orchestrator object."
                )

            if not "db_jobs" in db_config:
                raise SmartSimError(
                    err_message + "Could not find database job objects."
                )

            if not "steps" in db_config:
                raise SmartSimError(
                    err_message + "Could not find database job objects."
                )
            orc: Orchestrator = db_config["db"]

            # TODO check that each db_object is running

            job_steps = zip(db_config["db_jobs"].values(), db_config["steps"])
            try:
                for db_job, step in job_steps:
                    self._jobs.db_jobs[db_job.ename] = db_job
                    self._launcher.step_mapping[db_job.name] = step
                    if step.task_id:
                        self._launcher.task_manager.add_existing(int(step.task_id))
            except LauncherError as e:
                raise SmartSimError("Failed to reconnect orchestrator") from e

            # start job manager if not already started
            if not self._jobs.actively_monitoring:
                self._jobs.start()

            return orc

    def _set_dbobjects(self, manifest: Manifest) -> None:
        if not manifest.has_db_objects:
            return

        address_dict = self._jobs.get_db_host_addresses()
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
            logger.debug("Telemetry monitor started")


class _AnonymousBatchJob(EntityList[Model]):
    @staticmethod
    def _validate(model: Model) -> None:
        if model.batch_settings is None:
            msg = "Unable to create _AnonymousBatchJob without batch_settings"
            raise SmartSimError(msg)

    def __init__(self, model: Model) -> None:
        self._validate(model)
        super().__init__(model.name, model.path)
        self.entities = [model]
        self.batch_settings = model.batch_settings

    def _initialize_entities(self, **kwargs: t.Any) -> None: ...


def _look_up_launched_data(
    launcher: Launcher,
) -> t.Callable[[t.Tuple[str, Step]], "TStepLaunchMetaData"]:
    def _unpack_launched_data(data: t.Tuple[str, Step]) -> "TStepLaunchMetaData":
        # NOTE: we cannot assume that the name of the launched step
        # ``launched_step_name`` is equal to the name of the step referring to
        # the entity ``step.name`` as is the case when an entity list is
        # launched as a batch job
        launched_step_name, step = data
        launched_step_map = launcher.step_mapping[launched_step_name]
        out_file, err_file = step.get_output_files()
        return (
            launched_step_map.step_id,
            launched_step_map.task_id,
            launched_step_map.managed,
            out_file,
            err_file,
            pathlib.Path(step.meta.get("status_dir", step.cwd)),
        )

    return _unpack_launched_data
