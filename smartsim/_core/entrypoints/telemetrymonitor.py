# BSD 2-Clause License
#
# Copyright (c) 2021-2023 Hewlett Packard Enterprise
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

import argparse
import json
import logging
import os
import pathlib
import signal
import sys
import threading
import time
import typing as t

from dataclasses import dataclass, field
from types import FrameType

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.events import PatternMatchingEventHandler, LoggingEventHandler
from watchdog.events import FileCreatedEvent, FileModifiedEvent

from smartsim._core.config import CONFIG
from smartsim._core.control.job import JobEntity, _JobKey
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.launcher.stepInfo import StepInfo


from smartsim._core.launcher.cobalt.cobaltLauncher import CobaltLauncher
from smartsim._core.launcher.launcher import Launcher
from smartsim._core.launcher.local.local import LocalLauncher
from smartsim._core.launcher.lsf.lsfLauncher import LSFLauncher
from smartsim._core.launcher.pbs.pbsLauncher import PBSLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher
from smartsim._core.utils.helpers import get_ts
from smartsim._core.utils.serialize import TELMON_SUBDIR, MANIFEST_FILENAME

from smartsim.error.errors import SmartSimError
from smartsim.status import TERMINAL_STATUSES


"""
Telemetry Monitor entrypoint
"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
_EventClass = t.Literal["start", "stop", "timestep"]


@dataclass
class Run:
    """Model containing entities of an individual start call for an experiment"""

    timestamp: int
    models: t.List[JobEntity]
    orchestrators: t.List[JobEntity]
    ensembles: t.List[JobEntity]

    def flatten(
        self, filter_fn: t.Optional[t.Callable[[JobEntity], bool]] = None
    ) -> t.List[JobEntity]:
        """Flatten runs into a list of SmartSimEntity run events"""
        entities = self.models + self.orchestrators + self.ensembles
        if filter_fn:
            entities = [entity for entity in entities if filter_fn(entity)]
        return entities


@dataclass
class RuntimeManifest:
    """The runtime manifest holds meta information about the experiment entities created
    at runtime to satisfy the experiment requirements."""

    name: str
    path: pathlib.Path
    launcher: str
    runs: t.List[Run] = field(default_factory=list)


def _hydrate_persistable(
    persistable_entity: t.Dict[str, t.Any],
    entity_type: str,
    exp_dir: str,
) -> JobEntity:
    """Populate JobEntity instance with supplied metdata and instance details"""
    entity = JobEntity()

    metadata = persistable_entity["telemetry_metadata"]
    status_dir = pathlib.Path(metadata.get("status_dir"))

    entity.type = entity_type
    entity.name = persistable_entity["name"]
    entity.step_id = str(metadata.get("step_id") or "")
    entity.task_id = str(metadata.get("task_id") or "")
    entity.timestamp = int(persistable_entity.get("timestamp", "0"))
    entity.path = str(exp_dir)
    entity.status_dir = str(status_dir)

    return entity


def hydrate_persistable(
    entity_type: str,
    persistable_entity: t.Dict[str, t.Any],
    exp_dir: pathlib.Path,
) -> t.List[JobEntity]:
    """Map entity data persisted in a manifest file to an object"""
    entities = []

    # an entity w/parent key creates persistables for entities it contains
    parent_keys = {"shards", "models"}
    parent_keys = parent_keys.intersection(persistable_entity.keys())
    if parent_keys:
        container = "shards" if "shards" in parent_keys else "models"
        child_type = "orchestrator" if container == "shards" else "model"
        for child_entity in persistable_entity[container]:
            entity = _hydrate_persistable(child_entity, child_type, str(exp_dir))
            entities.append(entity)

        return entities

    entity = _hydrate_persistable(persistable_entity, entity_type, str(exp_dir))
    entities.append(entity)
    return entities


def hydrate_persistables(
    entity_type: str,
    run: t.Dict[str, t.Any],
    exp_dir: pathlib.Path,
) -> t.Dict[str, t.List[JobEntity]]:
    """Map a collection of entity data persisted in a manifest file to an object"""
    persisted: t.Dict[str, t.List[JobEntity]] = {
        "model": [],
        "orchestrator": [],
    }
    for item in run[entity_type]:
        entities = hydrate_persistable(entity_type, item, exp_dir)
        for new_entity in entities:
            persisted[new_entity.type].append(new_entity)

    return persisted


def hydrate_runs(
    persisted_runs: t.List[t.Dict[str, t.Any]], exp_dir: pathlib.Path
) -> t.List[Run]:
    """Map run data persisted in a manifest file to an object"""
    the_runs: t.List[Run] = []
    for run_instance in persisted_runs:
        run_entities: t.Dict[str, t.List[JobEntity]] = {
            "model": [],
            "orchestrator": [],
            "ensemble": [],
        }

        for key in run_entities:
            _entities = hydrate_persistables(key, run_instance, exp_dir)
            for entity_type, new_entities in _entities.items():
                if new_entities:
                    run_entities[entity_type].extend(new_entities)

        run = Run(
            run_instance["timestamp"],
            run_entities["model"],
            run_entities["orchestrator"],
            run_entities["ensemble"],
        )
        the_runs.append(run)

    return the_runs


def load_manifest(file_path: str) -> t.Optional[RuntimeManifest]:
    """Load a persisted manifest and return the content"""
    manifest_dict: t.Optional[t.Dict[str, t.Any]] = None
    try_count = 1

    while manifest_dict is None and try_count < 6:
        source = pathlib.Path(file_path)
        source = source.resolve()

        try:
            if text := source.read_text(encoding="utf-8").strip():
                manifest_dict = json.loads(text)
        except json.JSONDecodeError as ex:
            print(f"Error loading manifest: {ex}")
            # hack/fix: handle issues reading file before it is fully written
            time.sleep(0.5 * try_count)
        finally:
            try_count += 1

    if not manifest_dict:
        return None

    exp = manifest_dict.get("experiment", None)
    if not exp:
        raise ValueError("Manifest missing required experiment")

    runs = manifest_dict.get("runs", None)
    if runs is None:
        raise ValueError("Manifest missing required runs")

    exp_dir = pathlib.Path(exp["path"])
    runs = hydrate_runs(runs, exp_dir)

    manifest = RuntimeManifest(
        name=exp["name"],
        path=exp_dir,
        launcher=exp["launcher"],
        runs=runs,
    )
    return manifest


def track_event(
    timestamp: int,
    task_id: t.Union[int, str],
    step_id: str,
    etype: str,
    action: _EventClass,
    status_dir: pathlib.Path,
    logger: logging.Logger,
    detail: str = "",
    return_code: t.Optional[int] = None,
) -> None:
    """
    Persist a tracking event for an entity
    """
    tgt_path = status_dir / f"{action}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        task_id = int(task_id)
    except ValueError:
        pass

    entity_dict = {
        "timestamp": timestamp,
        "job_id": task_id,
        "step_id": step_id,
        "type": etype,
        "action": action,
    }

    if detail is not None:
        entity_dict["detail"] = detail

    if return_code is not None:
        entity_dict["return_code"] = return_code

    try:
        if not tgt_path.exists():
            # Don't overwrite existing tracking files
            bytes_written = tgt_path.write_text(json.dumps(entity_dict, indent=2))
            if bytes_written < 1:
                logger.warning("event tracking failed to write tracking file.")
    except Exception:
        logger.error("Unable to write tracking file.", exc_info=True)


class ManifestEventHandler(PatternMatchingEventHandler):
    """The ManifestEventHandler monitors an experiment for changes and updates
    a telemetry datastore as needed.

    It contains event handlers that are triggered by changes to a runtime experiment
    manifest. The runtime manifest differs from a standard manifest. A runtime manifest
    may contain multiple experiment executions in a `runs` collection.

    It also contains a long-polling loop that checks experiment entities for updates
    at each timestep."""

    def __init__(
        self,
        pattern: str,
        logger: logging.Logger,
        ignore_patterns: t.Any = None,
        ignore_directories: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(
            [pattern], ignore_patterns, ignore_directories, case_sensitive
        )  # type: ignore
        self._logger = logger
        self._tracked_runs: t.Dict[int, Run] = {}
        self._tracked_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._completed_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._launcher: t.Optional[Launcher] = None
        self.job_manager: JobManager = JobManager(threading.RLock())
        self._launcher_map: t.Dict[str, t.Type[Launcher]] = {
            "slurm": SlurmLauncher,
            "pbs": PBSLauncher,
            "cobalt": CobaltLauncher,
            "lsf": LSFLauncher,
            "local": LocalLauncher,
        }

    def init_launcher(self, launcher: str) -> Launcher:
        """Initialize the controller with a specific type of launcher.
        SmartSim currently supports slurm, pbs(pro), cobalt, lsf,
        and local launching

        :param launcher: which launcher to initialize
        :type launcher: str
        :raises SSUnsupportedError: if a string is passed that is not
                                    a supported launcher
        :raises TypeError: if no launcher argument is provided.
        """
        if not launcher:
            raise TypeError("Must provide a 'launcher' argument")

        if launcher_type := self._launcher_map.get(launcher.lower(), None):
            return launcher_type()

        raise ValueError("Launcher type not supported: " + launcher)

    def set_launcher(self, launcher_type: str) -> None:
        """Set the launcher for the experiment"""
        self._launcher = self.init_launcher(launcher_type)
        self.job_manager.set_launcher(self._launcher)
        self.job_manager.start()

    def process_manifest(self, manifest_path: str) -> None:
        """Read the runtime manifest for the experiment and track new entities

        :param manifest_path: The full path to the manifest file
        :type manifest_path: str"""
        try:
            manifest = load_manifest(manifest_path)
            if not manifest:
                return
        except json.JSONDecodeError:
            self._logger.error(f"Malformed manifest encountered: {manifest_path}")
            return
        except ValueError:
            self._logger.error("Manifest content error", exc_info=True)
            return

        self.set_launcher(manifest.launcher)

        if not self._launcher:
            raise SmartSimError(f"Unable to set launcher from {manifest_path}")

        runs = [run for run in manifest.runs if run.timestamp not in self._tracked_runs]

        exp_dir = pathlib.Path(manifest_path).parent.parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs and e.is_managed
            ):
                entity.path = str(exp_dir)

                self._tracked_jobs[entity.key] = entity
                track_event(
                    run.timestamp,
                    entity.task_id,
                    entity.step_id,
                    entity.type,
                    "start",
                    pathlib.Path(entity.status_dir),
                    self._logger,
                )

                if entity.is_managed:
                    self.job_manager.add_job(
                        entity.name,
                        entity.task_id,
                        entity,
                        False,
                    )
                    self._launcher.step_mapping.add(
                        entity.name, entity.step_id, entity.task_id, True
                    )
            self._tracked_runs[run.timestamp] = run

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Event handler for when a file or directory is modified.

        :param event: Event representing file/directory modification.
        :type event: FileModifiedEvent"""
        super().on_modified(event)  # type: ignore
        self._logger.info(f"processing manifest modified @ {event.src_path}")
        self.process_manifest(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event: Event representing file/directory creation.
        :type event: FileCreatedEvent"""
        super().on_created(event)  # type: ignore
        self._logger.info(f"processing manifest created @ {event.src_path}")
        self.process_manifest(event.src_path)

    def _to_completed(
        self,
        timestamp: int,
        entity: JobEntity,
        step_info: t.Optional[StepInfo],
    ) -> None:
        """Move a monitored entity from the active to completed collection to
        stop monitoring for updates during timesteps.

        :param timestamp: the current timestamp for event logging
        :type timestamp: int
        :param entity: the running SmartSim Job
        :type entity: JobEntity
        :param experiment_dir: the experiement directory to monitor for changes
        :type experiment_dir: pathlib.Path
        :param entity: the StepInfo received when requesting a Job status update
        :type entity: t.Optional[StepInfo]
        """
        inactive_entity = self._tracked_jobs.pop(entity.key)
        if entity.key not in self._completed_jobs:
            self._completed_jobs[entity.key] = inactive_entity

        job = self.job_manager[entity.name]
        self.job_manager.move_to_completed(job)

        if step_info:
            detail = f"status: {step_info.status}, error: {step_info.error}"
        else:
            detail = "unknown status. step_info not retrieved"

        if hasattr(job.entity, "status_dir"):
            write_path = pathlib.Path(job.entity.status_dir)

        track_event(
            timestamp,
            entity.task_id,
            entity.step_id,
            entity.type,
            "stop",
            write_path,
            self._logger,
            detail=detail,
        )

    def on_timestep(self, timestamp: int) -> None:
        """Called at polling frequency .to request status updates on
        monitored entities

        :param timestamp: the current timestamp for event logging
        :type timestamp: int
        :param experiment_dir: the experiement directory to monitor for changes
        :type experiment_dir: pathlib.Path"""
        entity_map = self._tracked_jobs

        if not self._launcher:
            return

        # consider not using name to avoid collisions
        names = {entity.name: entity for entity in entity_map.values()}

        if names:
            step_updates = self._launcher.get_step_update(list(names.keys()))

            for step_name, step_info in step_updates:
                if step_info and step_info.status in TERMINAL_STATUSES:
                    completed_entity = names[step_name]
                    self._to_completed(
                        timestamp, completed_entity, step_info
                    )


def can_shutdown(action_handler: ManifestEventHandler) -> bool:
    has_jobs = bool(action_handler.job_manager.jobs)
    has_dbs = bool(action_handler.job_manager.db_jobs)
    has_running_jobs = has_jobs or has_dbs

    return not has_running_jobs


def event_loop(
    observer: BaseObserver,
    action_handler: ManifestEventHandler,
    frequency: t.Union[int, float],
    logger: logging.Logger,
    cooldown_duration: int
) -> None:
    """Executes all attached timestep handlers every <frequency> seconds

    :param observer: (optional) a preconfigured watchdog Observer to inject
    :type observer: t.Optional[BaseObserver]
    :param action_handler: The manifest event processor instance
    :type action_handler: ManifestEventHandler
    :param frequency: frequency (in seconds) of update loop
    :type frequency: t.Union[int, float]
    :param experiment_dir: the experiement directory to monitor for changes
    :type experiment_dir: pathlib.Path
    :param logger: a preconfigured Logger instance
    :type logger: logging.Logger"""
    elapsed: int = 0
    last_ts: int = get_ts()

    while observer.is_alive():
        timestamp = get_ts()
        logger.debug(f"Telemetry timestep: {timestamp}")
        action_handler.on_timestep(timestamp)
        time.sleep(frequency)

        elapsed += timestamp - last_ts
        last_ts = timestamp

        if can_shutdown(action_handler):
            if elapsed >= cooldown_duration:
                observer.stop()  # type: ignore
        else:
            # reset cooldown any time there are still jobs running
            elapsed = 0


def main(
    frequency: t.Union[int, float],
    experiment_dir: pathlib.Path,
    logger: logging.Logger,
    observer: t.Optional[BaseObserver] = None,
) -> int:
    """Setup the monitoring entities and start the timer-based loop that
    will poll for telemetry data

    :param frequency: frequency (in seconds) of update loop
    :type frequency: t.Union[int, float]
    :param experiment_dir: the experiement directory to monitor for changes
    :type experiment_dir: pathlib.Path
    :param observer: (optional) a preconfigured Observer to inject
    :type observer: t.Optional[BaseObserver]
    """
    manifest_relpath = pathlib.Path(TELMON_SUBDIR) / MANIFEST_FILENAME
    manifest_path = experiment_dir / manifest_relpath
    monitor_pattern = str(manifest_relpath)

    logger.info(
        f"Executing telemetry monitor with frequency: {frequency}s"
        f", on target directory: {experiment_dir}"
        f" matching pattern: {monitor_pattern}"
    )

    telemetry_cooldown = CONFIG.telemetry_cooldown
    log_handler = LoggingEventHandler(logger)  # type: ignore
    action_handler = ManifestEventHandler(monitor_pattern, logger)

    if observer is None:
        observer = Observer()

    try:
        if manifest_path.exists():
            # a manifest may not exist depending on startup timing
            action_handler.process_manifest(str(manifest_path))

        observer.schedule(log_handler, experiment_dir, recursive=True)  # type:ignore
        observer.schedule(action_handler, experiment_dir, recursive=True)  # type:ignore
        observer.start()  # type: ignore

        event_loop(observer, action_handler, frequency, logger, telemetry_cooldown)
        return 0
    except Exception as ex:
        logger.error(ex)
    finally:
        if observer.is_alive():
            observer.stop()  # type: ignore
            observer.join()

    return 1


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
        logger = logging.getLogger()
        logger.warning("Received signal with no signo")


def register_signal_handlers() -> None:
    """Register a signal handling function for all termination events"""
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments"""
    arg_parser = argparse.ArgumentParser(description="SmartSim Telemetry Monitor")
    arg_parser.add_argument(
        "-frequency",
        type=str,
        help="Frequency of telemetry updates (in seconds))",
        required=True,
    )
    arg_parser.add_argument(
        "-exp_dir",
        type=str,
        help="Experiment root directory",
        required=True,
    )
    return arg_parser


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = get_parser()
    args = parser.parse_args()

    log = logging.getLogger()

    # Must register cleanup before the main loop is running
    register_signal_handlers()

    try:
        main(int(args.frequency), pathlib.Path(args.exp_dir), log)
        sys.exit(0)
    except Exception:
        log.exception(
            "Shutting down telemetry monitor due to unexpected error", exc_info=True
        )

    sys.exit(1)
