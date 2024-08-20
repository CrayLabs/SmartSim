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
import os
import os.path as osp
import pathlib
import pickle
import signal
import subprocess
import sys
import threading
import time
import typing as t

from smartsim._core.utils.network import get_ip_from_host
from smartsim.entity._mock import Mock

from ..._core.launcher.step import Step
from ..._core.utils.helpers import (
    SignalInterceptionStack,
    unpack_colo_fs_identifier,
    unpack_fs_identifier,
)
from ...database import FeatureStore
from ...entity import Application, Ensemble, EntitySequence, SmartSimEntity
from ...error import (
    LauncherError,
    SmartSimError,
    SSDBIDConflictError,
    SSInternalError,
    SSUnsupportedError,
)
from ...log import get_logger
from ...servertype import CLUSTERED, STANDALONE
from ...status import TERMINAL_STATUSES, JobStatus
from ..config import CONFIG
from ..launcher import (
    DragonLauncher,
    LocalLauncher,
    LSFLauncher,
    PBSLauncher,
    SGELauncher,
    SlurmLauncher,
)
from ..launcher.launcher import Launcher
from ..utils import serialize
from .controller_utils import _AnonymousBatchJob, _look_up_launched_data
from .job import Job
from .jobmanager import JobManager
from .manifest import LaunchedManifest, LaunchedManifestBuilder, Manifest

if t.TYPE_CHECKING:
    from types import FrameType

    from ..utils.serialize import TStepLaunchMetaData


logger = get_logger(__name__)

# job manager lock
JM_LOCK = threading.RLock()


class Client(Mock):
    """Mock Client"""


class ConfigOptions(Mock):
    """Mock ConfigOptions"""


def fs_is_active():
    pass


def set_ml_model():
    pass


def set_script():
    pass


def shutdown_fs_node():
    pass


def create_cluster():
    pass


def check_cluster_status():
    pass


class Controller:
    """The controller module provides an interface between the
    smartsim entities created in the experiment and the
    underlying workload manager or run framework.
    """

    def __init__(self, launcher: str = "local") -> None:
        """Initialize a Controller

        :param launcher: the type of launcher being used
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
        # launch a telemetry monitor to track job progress
        if CONFIG.telemetry_enabled:
            self._start_telemetry_monitor(exp_path)

        self._jobs.kill_on_interrupt = kill_on_interrupt

        # register custom signal handler for ^C (SIGINT)
        SignalInterceptionStack.get(signal.SIGINT).push_unique(
            self._jobs.signal_interrupt
        )
        launched = self._launch(exp_name, exp_path, manifest)

        # start the job manager thread if not already started
        if not self._jobs.actively_monitoring:
            self._jobs.start()

        serialize.save_launch_manifest(
            launched.map(_look_up_launched_data(self._launcher))
        )

        # block until all non-feature store jobs are complete
        if block:
            # poll handles its own keyboard interrupt as
            # it may be called separately
            self.poll(5, True, kill_on_interrupt=kill_on_interrupt)

    @property
    def active_feature_store_jobs(self) -> t.Dict[str, Job]:
        """Return active feature store jobs."""
        return {**self._jobs.fs_jobs}

    @property
    def feature_store_active(self) -> bool:
        with JM_LOCK:
            if len(self._jobs.fs_jobs) > 0:
                return True
            return False

    def poll(
        self, interval: int, verbose: bool, kill_on_interrupt: bool = True
    ) -> None:
        """Poll running jobs and receive logging output of job status

        :param interval: number of seconds to wait before polling again
        :param verbose: set verbosity
        :param kill_on_interrupt: flag for killing jobs when SIGINT is received
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
        :returns: bool
        :raises ValueError: if entity has not been launched yet
        """
        try:
            if isinstance(entity, FeatureStore):
                raise TypeError("Finished() does not support FeatureStore instances")
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
        """
        with JM_LOCK:
            job = self._jobs[entity.name]
            if job.status not in TERMINAL_STATUSES:
                logger.info(
                    " ".join(
                        (
                            "Stopping application",
                            entity.name,
                            "with job name",
                            str(job.name),
                        )
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

    def stop_fs(self, fs: FeatureStore) -> None:
        """Stop an FeatureStore

        :param fs: FeatureStore to be stopped
        """
        if fs.batch:
            self.stop_entity(fs)
        else:
            with JM_LOCK:
                for node in fs.entities:
                    for host_ip, port in itertools.product(
                        (get_ip_from_host(host) for host in node.hosts), fs.ports
                    ):
                        retcode, _, _ = shutdown_fs_node(host_ip, port)
                        # Sometimes the fs will not shutdown (unless we force NOSAVE)
                        if retcode != 0:
                            self.stop_entity(node)
                            continue

                        job = self._jobs[node.name]
                        job.set_status(
                            JobStatus.CANCELLED,
                            "",
                            0,
                            output=None,
                            error=None,
                        )
                        self._jobs.move_to_completed(job)

        fs.reset_hosts()

    def stop_entity_list(self, entity_list: EntitySequence[SmartSimEntity]) -> None:
        """Stop an instance of an entity list

        :param entity_list: entity list to be stopped
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
    ) -> JobStatus:
        """Get the status of an entity

        :param entity: entity to get status of
        :raises TypeError: if not SmartSimEntity | EntitySequence
        :return: status of entity
        """
        if not isinstance(entity, (SmartSimEntity, EntitySequence)):
            raise TypeError(
                "Argument must be of type SmartSimEntity or EntitySequence, "
                f"not {type(entity)}"
            )
        return self._jobs.get_status(entity)

    def get_entity_list_status(
        self, entity_list: EntitySequence[SmartSimEntity]
    ) -> t.List[JobStatus]:
        """Get the statuses of an entity list

        :param entity_list: entity list containing entities to
                            get statuses of
        :raises TypeError: if not EntitySequence
        :return: list of SmartSimStatus statuses
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
            "dragon": DragonLauncher,
            "sge": SGELauncher,
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

    @staticmethod
    def symlink_output_files(
        job_step: Step, entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> None:
        """Create symlinks for entity output files that point to the output files
        under the .smartsim directory

        :param job_step: Job step instance
        :param entity: Entity instance
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
    ) -> LaunchedManifest[t.Tuple[str, Step]]:
        """Main launching function of the controller

        FeatureStores are always launched first so that the
        address of the feature store can be given to following entities

        :param exp_name: The name of the launching experiment
        :param exp_path: path to location of ``Experiment`` directory if generated
        :param manifest: Manifest of deployables to launch
        """

        manifest_builder = LaunchedManifestBuilder[t.Tuple[str, Step]](
            exp_name=exp_name,
            exp_path=exp_path,
            launcher_name=str(self._launcher),
        )
        # Loop over deployables to launch and launch multiple FeatureStores
        for featurestore in manifest.fss:
            for key in self._jobs.get_fs_host_addresses():
                _, fs_id = unpack_fs_identifier(key, "_")
                if featurestore.fs_identifier == fs_id:
                    raise SSDBIDConflictError(
                        f"Feature store identifier {featurestore.fs_identifier}"
                        " has already been used. Pass in a unique"
                        " name for fs_identifier"
                    )

            if featurestore.num_shards > 1 and isinstance(
                self._launcher, LocalLauncher
            ):
                raise SmartSimError(
                    "Local launcher does not support multi-host feature stores"
                )
            self._launch_feature_store(featurestore, manifest_builder)

        if self.feature_store_active:
            self._set_fsobjects(manifest)

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
                for substep, substep_entity in zip(substeps, elist.applications):
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
        # applications themselves cannot be batch steps. If batch settings are
        # attached, wrap them in an anonymous batch job step
        for application in manifest.applications:
            application_telem_dir = (
                manifest_builder.run_telemetry_subdirectory / "application"
            )
            if application.batch_settings:
                anon_entity_list = _AnonymousBatchJob(application)
                batch_step, substeps = self._create_batch_job_step(
                    anon_entity_list, application_telem_dir
                )
                manifest_builder.add_application(
                    application, (batch_step.name, batch_step)
                )

                symlink_substeps.append((substeps[0], application))
                steps.append((batch_step, application))
            else:
                # create job step for aapplication with run settings
                job_step = self._create_job_step(application, application_telem_dir)
                manifest_builder.add_application(application, (job_step.name, job_step))
                steps.append((job_step, application))

        # launch and symlink steps
        for step, entity in steps:
            self._launch_step(step, entity)
            self.symlink_output_files(step, entity)

        # symlink substeps to maintain directory structure
        for substep, entity in symlink_substeps:
            self.symlink_output_files(substep, entity)

        return manifest_builder.finalize()

    def _launch_feature_store(
        self,
        featurestore: FeatureStore,
        manifest_builder: LaunchedManifestBuilder[t.Tuple[str, Step]],
    ) -> None:
        """Launch an FeatureStore instance

        This function will launch the FeatureStore instance and
        if on WLM, find the nodes where it was launched and
        set them in the JobManager

        :param featurestore: FeatureStore to launch
        :param manifest_builder: An `LaunchedManifestBuilder` to record the
                                 names and `Step`s of the launched featurestore
        """
        featurestore.remove_stale_files()
        feature_store_telem_dir = (
            manifest_builder.run_telemetry_subdirectory / "database"
        )

        # if the featurestore was launched as a batch workload
        if featurestore.batch:
            feature_store_batch_step, substeps = self._create_batch_job_step(
                featurestore, feature_store_telem_dir
            )
            manifest_builder.add_feature_store(
                featurestore,
                [(feature_store_batch_step.name, step) for step in substeps],
            )

            self._launch_step(feature_store_batch_step, featurestore)
            self.symlink_output_files(feature_store_batch_step, featurestore)

            # symlink substeps to maintain directory structure
            for substep, substep_entity in zip(substeps, featurestore.entities):
                self.symlink_output_files(substep, substep_entity)

        # if featurestore was run on existing allocation, locally, or in allocation
        else:
            fs_steps = [
                (
                    self._create_job_step(
                        fs, feature_store_telem_dir / featurestore.name
                    ),
                    fs,
                )
                for fs in featurestore.entities
            ]
            manifest_builder.add_feature_store(
                featurestore, [(step.name, step) for step, _ in fs_steps]
            )
            for fs_step in fs_steps:
                self._launch_step(*fs_step)
                self.symlink_output_files(*fs_step)

        # wait for featurestore to spin up
        self._feature_store_launch_wait(featurestore)

        # set the jobs in the job manager to provide SSDB variable to entities
        # if _host isnt set within each
        self._jobs.set_fs_hosts(featurestore)

        # create the feature store cluster
        if featurestore.num_shards > 2:
            num_trials = 5
            cluster_created = False
            while not cluster_created:
                try:
                    create_cluster(featurestore.hosts, featurestore.ports)
                    check_cluster_status(featurestore.hosts, featurestore.ports)
                    num_shards = featurestore.num_shards
                    logger.info(
                        f"Feature store cluster created with {num_shards} shards"
                    )
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
        self._save_feature_store(featurestore)
        logger.debug(f"FeatureStore launched on nodes: {featurestore.hosts}")

    def _launch_step(
        self,
        job_step: Step,
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]],
    ) -> None:
        """Use the launcher to launch a job step

        :param job_step: a job step instance
        :param entity: entity instance
        :raises SmartSimError: if launch fails
        """
        # attempt to retrieve entity name in JobManager.completed
        completed_job = self._jobs.completed.get(entity.name, None)

        # if completed job DNE and is the entity name is not
        # running in JobManager.jobs or JobManager.fs_jobs,
        # launch the job
        if completed_job is None and (
            entity.name not in self._jobs.jobs and entity.name not in self._jobs.fs_jobs
        ):
            try:
                job_id = self._launcher.run(job_step)
            except LauncherError as e:
                msg = f"An error occurred when launching {entity.name} \n"
                msg += "Check error and output files for details.\n"
                msg += f"{entity}"
                logger.error(msg)
                raise SmartSimError(f"Job step {entity.name} failed to launch") from e

        # if the completed job does exist and the entity passed in is the same
        # that has ran and completed, relaunch the entity.
        elif completed_job is not None and completed_job.entity is entity:
            try:
                job_id = self._launcher.run(job_step)
            except LauncherError as e:
                msg = f"An error occurred when launching {entity.name} \n"
                msg += "Check error and output files for details.\n"
                msg += f"{entity}"
                logger.error(msg)
                raise SmartSimError(f"Job step {entity.name} failed to launch") from e

        # the entity is using a duplicate name of an existing entity in
        # the experiment, throw an error
        else:
            raise SSUnsupportedError("SmartSim entities cannot have duplicate names.")

        # a job step is a task if it is not managed by a workload manager (i.e. Slurm)
        # but is rather started, monitored, and exited through the Popen interface
        # in the taskmanager
        is_task = not job_step.managed

        if self._jobs.query_restart(entity.name):
            logger.debug(f"Restarting {entity.name}")
            self._jobs.restart_job(job_step.name, job_id, entity.name, is_task)
        else:
            logger.debug(f"Launching {entity.name}")
            self._jobs.add_job(job_step, job_id, is_task)

    def _create_batch_job_step(
        self,
        entity_list: t.Union[FeatureStore, Ensemble, _AnonymousBatchJob],
        telemetry_dir: pathlib.Path,
    ) -> t.Tuple[Step, t.List[Step]]:
        """Use launcher to create batch job step

        :param entity_list: EntityList to launch as batch
        :param telemetry_dir: Path to a directory in which the batch job step
                              may write telemetry events
        :return: batch job step instance and a list of run steps to be
                 executed within the batch job
        """
        if not entity_list.batch_settings:
            raise ValueError(
                "EntityList must have batch settings to be launched as batch"
            )

        telemetry_dir = telemetry_dir / entity_list.name
        batch_step = self._launcher.create_step(entity, entity_list.batch_settings)
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
        :param telemetry_dir: Path to a directory in which the job step
                               may write telemetry events
        :return: the job step
        """
        # get SSDB, SSIN, SSOUT and add to entity run settings
        if isinstance(entity, Application):
            self._prep_entity_client_env(entity)

        # creating job step through the created launcher
        step = self._launcher.create_step(entity, entity.run_settings)

        step.meta["entity_type"] = str(type(entity).__name__).lower()
        step.meta["status_dir"] = str(telemetry_dir / entity.name)

        # return the job step that was created using the launcher since the launcher is defined in the exp
        return step

    def _prep_entity_client_env(self, entity: Application) -> None:
        """Retrieve all connections registered to this entity

        :param entity: The entity to retrieve connections from
        """
        client_env: t.Dict[str, t.Union[str, int, float, bool]] = {}
        address_dict = self._jobs.get_fs_host_addresses()

        for fs_id, addresses in address_dict.items():
            fs_name, _ = unpack_fs_identifier(fs_id, "_")
            if addresses:
                # Cap max length of SSDB
                client_env[f"SSDB{fs_name}"] = ",".join(addresses[:128])

                # Retrieve num_shards to append to client env
                client_env[f"SR_fs_TYPE{fs_name}"] = (
                    CLUSTERED if len(addresses) > 1 else STANDALONE
                )

        if entity.incoming_entities:
            client_env["SSKEYIN"] = ",".join(
                [in_entity.name for in_entity in entity.incoming_entities]
            )
        if entity.query_key_prefixing():
            client_env["SSKEYOUT"] = entity.name

        # Set address to local if it's a colocated application
        if entity.colocated and entity.run_settings.colocated_fs_settings is not None:
            fs_name_colo = entity.run_settings.colocated_fs_settings["fs_identifier"]
            assert isinstance(fs_name_colo, str)
            for key in address_dict:
                _, fs_id = unpack_fs_identifier(key, "_")
                if fs_name_colo == fs_id:
                    raise SSDBIDConflictError(
                        f"Feature store identifier {fs_name_colo}"
                        " has already been used. Pass in a unique"
                        " name for fs_identifier"
                    )

            fs_name_colo = unpack_colo_fs_identifier(fs_name_colo)
            if colo_cfg := entity.run_settings.colocated_fs_settings:
                port = colo_cfg.get("port", None)
                socket = colo_cfg.get("unix_socket", None)
                if socket and port:
                    raise SSInternalError(
                        "Co-located was configured for both TCP/IP and UDS"
                    )
                if port:
                    client_env[f"SSDB{fs_name_colo}"] = f"127.0.0.1:{str(port)}"
                elif socket:
                    client_env[f"SSDB{fs_name_colo}"] = f"unix://{socket}"
                else:
                    raise SSInternalError(
                        "Colocated feature store was not configured for either TCP or UDS"
                    )
                client_env[f"SR_fs_TYPE{fs_name_colo}"] = STANDALONE
        entity.run_settings.update_env(client_env)

    def _save_feature_store(self, feature_store: FeatureStore) -> None:
        """Save the FeatureStore object via pickle

        This function saves the feature store information to a pickle
        file that can be imported by subsequent experiments to reconnect
        to the featurestore.

        :param featurestore: FeatureStore configuration to be saved
        """

        if not feature_store.is_active():
            raise Exception("Feature store is not running")

        # Extract only the fs_jobs associated with this particular feature store
        if feature_store.batch:
            job_names = [feature_store.name]
        else:
            job_names = [fsnode.name for fsnode in feature_store.entities]
        fs_jobs = {
            name: job for name, job in self._jobs.fs_jobs.items() if name in job_names
        }

        # Extract the associated steps
        steps = [
            self._launcher.step_mapping[fs_job.name] for fs_job in fs_jobs.values()
        ]

        feature_store_data = {"fs": feature_store, "fs_jobs": fs_jobs, "steps": steps}

        with open(feature_store.checkpoint_file, "wb") as pickle_file:
            pickle.dump(feature_store_data, pickle_file)

        # Extract only the fs_jobs associated with this particular featurestore
        if feature_store.batch:
            job_names = [feature_store.name]
        else:
            job_names = [fsnode.name for fsnode in feature_store.entities]
        fs_jobs = {
            name: job for name, job in self._jobs.fs_jobs.items() if name in job_names
        }

        # Extract the associated steps
        steps = [
            self._launcher.step_mapping[fs_job.name] for fs_job in fs_jobs.values()
        ]

        feature_store_data = {"fs": feature_store, "fs_jobs": fs_jobs, "steps": steps}

        with open(feature_store.checkpoint_file, "wb") as pickle_file:
            pickle.dump(feature_store_data, pickle_file)

    def _feature_store_launch_wait(self, featurestore: FeatureStore) -> None:
        """Wait for the featurestore instances to run

        In the case where the featurestore is launched as a batch
        through a WLM, we wait for the featurestore to exit the
        queue before proceeding so new launched entities can
        be launched with SSDB address

        :param featurestore: FeatureStore instance
        :raises SmartSimError: if launch fails or manually stopped by user
        """
        if featurestore.batch:
            logger.info("FeatureStore launched as a batch")
            logger.info("While queued, SmartSim will wait for FeatureStore to run")
            logger.info("CTRL+C interrupt to abort and cancel launch")

        ready = False
        while not ready:
            try:
                time.sleep(CONFIG.jm_interval)
                # manually trigger job update if JM not running
                if not self._jobs.actively_monitoring:
                    self._jobs.check_jobs()

                # _jobs.get_status acquires JM lock for main thread, no need for locking
                statuses = self.get_entity_list_status(featurestore)
                if all(stat == JobStatus.RUNNING for stat in statuses):
                    ready = True
                    # TODO: Add a node status check
                elif any(stat in TERMINAL_STATUSES for stat in statuses):
                    self.stop_fs(featurestore)
                    msg = "FeatureStore failed during startup"
                    msg += f" See {featurestore.path} for details"
                    raise SmartSimError(msg)
                else:
                    logger.debug("Waiting for featurestore instances to spin up...")
            except KeyboardInterrupt:
                logger.info("FeatureStore launch cancelled - requesting to stop")
                self.stop_fs(featurestore)

                # re-raise keyboard interrupt so the job manager will display
                # any running and un-killed jobs as this method is only called
                # during launch and we handle all keyboard interrupts during
                # launch explicitly
                raise

    def reload_saved_fs(
        self, checkpoint_file: t.Union[str, os.PathLike[str]]
    ) -> FeatureStore:
        with JM_LOCK:

            if not osp.exists(checkpoint_file):
                raise FileNotFoundError(
                    f"The SmartSim feature store config file {os.fspath(checkpoint_file)} "
                    "cannot be found."
                )

            try:
                with open(checkpoint_file, "rb") as pickle_file:
                    fs_config = pickle.load(pickle_file)
            except (OSError, IOError) as e:
                msg = "Feature store checkpoint corrupted"
                raise SmartSimError(msg) from e

            err_message = (
                "The SmartSim feature store checkpoint is incomplete or corrupted. "
            )
            if not "fs" in fs_config:
                raise SmartSimError(
                    err_message + "Could not find the featurestore object."
                )

            if not "fs_jobs" in fs_config:
                raise SmartSimError(
                    err_message + "Could not find feature store job objects."
                )

            if not "steps" in fs_config:
                raise SmartSimError(
                    err_message + "Could not find feature store job objects."
                )
            feature_store: FeatureStore = fs_config["fs"]

            # TODO check that each fs_object is running

            job_steps = zip(fs_config["fs_jobs"].values(), fs_config["steps"])
            try:
                for fs_job, step in job_steps:
                    self._jobs.fs_jobs[fs_job.ename] = fs_job
                    self._launcher.add_step_to_mapping_table(fs_job.name, step)
                    if step.task_id:
                        self._launcher.task_manager.add_existing(int(step.task_id))
            except LauncherError as e:
                raise SmartSimError("Failed to reconnect feature store") from e

            # start job manager if not already started
            if not self._jobs.actively_monitoring:
                self._jobs.start()

            return feature_store

    def _set_fsobjects(self, manifest: Manifest) -> None:
        if not manifest.has_fs_objects:
            return

        address_dict = self._jobs.get_fs_host_addresses()
        for (
            fs_id,
            fs_addresses,
        ) in address_dict.items():
            fs_name, name = unpack_fs_identifier(fs_id, "_")

            hosts = list({address.split(":")[0] for address in fs_addresses})
            ports = list({int(address.split(":")[-1]) for address in fs_addresses})

            if not fs_is_active(hosts=hosts, ports=ports, num_shards=len(fs_addresses)):
                raise SSInternalError("Cannot set FS Objects, FS is not running")

            os.environ[f"SSDB{fs_name}"] = fs_addresses[0]

            os.environ[f"SR_fs_TYPE{fs_name}"] = (
                CLUSTERED if len(fs_addresses) > 1 else STANDALONE
            )

            options = ConfigOptions.create_from_environment(name)
            client = Client(options, logger_name="SmartSim")

            for application in manifest.applications:
                if not application.colocated:
                    for fs_model in application.fs_models:
                        set_ml_model(fs_model, client)
                    for fs_script in application.fs_scripts:
                        set_script(fs_script, client)

            for ensemble in manifest.ensembles:
                for fs_model in ensemble.fs_models:
                    set_ml_model(fs_model, client)
                for fs_script in ensemble.fs_scripts:
                    set_script(fs_script, client)
                for entity in ensemble.applications:
                    if not entity.colocated:
                        # Set models which could belong only
                        # to the entities and not to the ensemble
                        # but avoid duplicates
                        for fs_model in entity.fs_models:
                            if fs_model not in ensemble.fs_models:
                                set_ml_model(fs_model, client)
                        for fs_script in entity.fs_scripts:
                            if fs_script not in ensemble.fs_scripts:
                                set_script(fs_script, client)

    def _start_telemetry_monitor(self, exp_dir: str) -> None:
        """Spawns a telemetry monitor process to keep track of the life times
        of the processes launched through this controller.

        :param exp_dir: An experiment directory
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
