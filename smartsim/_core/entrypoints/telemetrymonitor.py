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
from datetime import datetime
from types import FrameType

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.events import PatternMatchingEventHandler, LoggingEventHandler
from watchdog.events import FileCreatedEvent, FileModifiedEvent

from smartsim._core.control.job import Job, JobEntity
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.launcher.stepInfo import StepInfo
from smartsim._core.launcher.cobalt.cobaltLauncher import CobaltLauncher

from smartsim._core.launcher.launcher import Launcher
from smartsim._core.launcher.local.local import LocalLauncher
from smartsim._core.launcher.lsf.lsfLauncher import LSFLauncher
from smartsim._core.launcher.pbs.pbsLauncher import PBSLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher

# from smartsim.entity import SmartSimEntity, EntityList
from smartsim.error.errors import SmartSimError
from smartsim.status import TERMINAL_STATUSES


logging.basicConfig(level=logging.INFO)

"""
Telemetry Monitor entrypoint
"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
_EventClass = t.Literal["start", "stop", "timestep"]
_ManifestKey = t.Literal["timestamp", "model", "orchestrator", "ensemble", "run_id"]
_JobKey = t.Tuple[str, str]


# class PersistableEntity(JobEntity):
#     """Minimal model required for monitoring an entity in the JobManager"""

#     def __init__(
#         self,
#         etype: str,
#         name: str,
#         job_id: str,
#         step_id: str,
#         timestamp: int,
#         exp_dir: str,
#     ) -> None:
#         super().__init__()
#         self.type = etype
#         self.name = name
#         self.job_id = job_id
#         self.step_id = step_id
#         self.path = exp_dir
#         self.timestamp = timestamp

#     @property
#     def key(self) -> _JobKey:
#         return (self.job_id, self.step_id)


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


def get_ts() -> int:
    """Helper function to ensure all timestamps are converted to integers"""
    return int(datetime.timestamp(datetime.now()))


def hydrate_persistable(
    entity_type: str,
    persisted_entity: t.Dict[str, t.Any],
    # timestamp: int,
    exp_dir: pathlib.Path,
) -> JobEntity:
    """Map entity data persisted in a manifest file to an object"""
    entity = JobEntity()

    entity.type = entity_type
    entity.name = persisted_entity["name"]
    entity.job_id = persisted_entity.get("job_id", "")
    entity.step_id = persisted_entity.get("step_id", "")
    entity.timestamp = int(persisted_entity.get("run_id", "0"))
    entity.path = str(exp_dir)

    return entity


def hydrate_persistables(
    entity_type: _ManifestKey,
    run: t.Dict[_ManifestKey, t.Any],
    exp_dir: pathlib.Path,
) -> t.List[JobEntity]:
    """Map a collection of entity data persisted in a manifest file to an object"""
    persisted: t.List[JobEntity] = []

    for item in run[entity_type]:
        entity = hydrate_persistable(entity_type, item, exp_dir)
        persisted.append(entity)

    return persisted


def hydrate_runs(
    persisted_runs: t.List[t.Dict[_ManifestKey, t.Any]], exp_dir: pathlib.Path
) -> t.List[Run]:
    """Map run data persisted in a manifest file to an object"""
    runs = [
        Run(
            timestamp=instance["run_id"],
            models=hydrate_persistables("model", instance, exp_dir),
            orchestrators=hydrate_persistables("orchestrator", instance, exp_dir),
            ensembles=hydrate_persistables("ensemble", instance, exp_dir),
        )
        for instance in persisted_runs
    ]
    return runs


def load_manifest(file_path: str) -> RuntimeManifest:
    """Load a persisted manifest and return the content"""
    source = pathlib.Path(file_path)
    source = source.resolve()

    text = source.read_text(encoding="utf-8")
    manifest_dict = json.loads(text)
    exp_dir = pathlib.Path(manifest_dict["experiment"]["path"])

    manifest = RuntimeManifest(
        name=manifest_dict["experiment"]["name"],
        path=exp_dir,
        launcher=manifest_dict["experiment"]["launcher"],
        runs=hydrate_runs(manifest_dict["runs"], exp_dir),
    )
    return manifest


def track_event(
    timestamp: int,
    # entity: JobEntity,
    ename: str,
    job_id: str,
    step_id: str,
    etype: str,
    action: _EventClass,
    exp_dir: pathlib.Path,
    logger: logging.Logger,
    detail: str = "",
) -> None:
    """
    Persist a tracking event for an entity
    """
    job_id = job_id or ""
    step_id = step_id or ""
    entity_type = etype or "missing_entity_type"
    logger.info(
        f"tracking `{entity_type}.{action}` event w/jid: {job_id}, "
        f"tid: {step_id}, ts: {timestamp}"
    )

    name: str = ename or "entity-name-not-found"
    tgt_path = exp_dir / "manifest" / entity_type / name / f"{action}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        entity_dict = {
            "job_id": job_id,
            "step_id": step_id,
            "type": etype,
            "action": action,
            # "path": str(exp_dir),
            "detail": detail,
        }
        # entity_dict.pop("path", None)
        # entity_dict["detail"] = detail
        tgt_path.write_text(json.dumps(entity_dict))
    except Exception:
        logger.error("Unable to write tracking file.", exc_info=True)


def track_completed(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for the end of job"""
    # inactive_entity = job.entity
    detail = job.status
    exp_dir = pathlib.Path(job.entity.path)

    # track_event(get_ts(), inactive_entity, "stop", exp_dir, logger, detail=detail)
    track_event(
        get_ts(),
        job.entity.name,
        "",
        job.jid or "",
        job.entity.type,
        "stop",
        exp_dir,
        logger,
        detail=detail,
    )


def track_started(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for the start of job"""
    # inactive_entity = job.entity
    exp_dir = pathlib.Path(job.entity.path)

    # track_event(get_ts(), inactive_entity, "start", exp_dir, logger)
    track_event(
        get_ts(),
        job.entity.name,
        "",
        job.jid or "",
        job.entity.type,
        "start",
        exp_dir,
        logger,
    )


def track_timestep(job: Job, logger: logging.Logger) -> None:
    """Persists telemetry event for a timestep"""
    # inactive_entity = job.entity
    exp_dir = pathlib.Path(job.entity.path)

    track_event(
        get_ts(),
        job.entity.name,
        "",
        job.jid or "",
        job.entity.type,
        "timestep",
        exp_dir,
        logger,
    )


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
        self._launcher_type: str = ""
        self._launcher: t.Optional[Launcher] = None
        self._jm: JobManager = JobManager(threading.RLock())
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

        if launcher is not None:
            launcher = launcher.lower()
            if launcher in self._launcher_map:
                return self._launcher_map[launcher]()
            raise ValueError("Launcher type not supported: " + launcher)

        raise TypeError("Must provide a 'launcher' argument")

    def set_launcher(self, launcher_type: str) -> None:
        """Set the launcher for the experiment"""
        if launcher_type != self._launcher_type:
            self._launcher_type = launcher_type
            self._launcher = self.init_launcher(launcher_type)
            self._jm.set_launcher(self._launcher)
            self._jm.add_job_onstart_callback(track_started)
            self._jm.add_job_onstop_callback(track_completed)
            self._jm.add_job_onstep_callback(track_timestep)
            self._jm.start()

    @property
    def launcher(self) -> Launcher:
        """Return a launcher appropriate for the experiment"""
        if not self._launcher:
            self._launcher = LocalLauncher()

        # if not self._launcher:
        #     raise ValueError("Launcher failed to instantiate properly")

        return self._launcher

    def process_manifest(self, manifest_path: str) -> None:
        """Load the runtime manifest for the experiment and add the entities
        to the collections of items being tracked for updates"""
        manifest = load_manifest(manifest_path)
        self.set_launcher(manifest.launcher)

        if not self._jm._launcher:  # pylint: disable=protected-access
            raise SmartSimError(f"Unable to set launcher from {manifest_path}")

        runs = [run for run in manifest.runs if run.timestamp not in self._tracked_runs]

        # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        exp_dir = pathlib.Path(manifest_path).parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs
            ):
                entity.path = str(exp_dir)

                self._tracked_jobs[entity.key] = entity
                track_event(
                    run.timestamp,
                    entity.name,
                    entity.job_id,
                    entity.step_id,
                    entity.type,
                    "start",
                    exp_dir,
                    self._logger,
                )

                self._jm.add_job(
                    entity.name,
                    entity.job_id,
                    entity,
                    entity.is_managed,
                    # is_orch=entity.is_db,
                )
                self._jm._launcher.step_mapping.add(  # pylint: disable=protected-access
                    entity.name, entity.step_id, entity.step_id, entity.is_managed
                )
            self._tracked_runs[run.timestamp] = run

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Event handler for when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """
        super().on_modified(event)  # type: ignore
        self.process_manifest(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
        super().on_created(event)  # type: ignore
        self.process_manifest(event.src_path)

    def _to_completed(
        self,
        timestamp: int,
        entity: JobEntity,
        exp_dir: pathlib.Path,
        step_info: t.Optional[StepInfo],
    ) -> None:
        """Move a monitored entity from the active to completed collection to
        stop monitoring for updates during timesteps."""
        inactive_entity = self._tracked_jobs.pop(entity.key)
        if entity.key not in self._completed_jobs:
            self._completed_jobs[entity.key] = inactive_entity

        job = self._jm[entity.name]
        self._jm.move_to_completed(job)

        if step_info:
            detail = f"status: {step_info.status}, error: {step_info.error}"
        else:
            detail = "unknown status. step_info not retrieved"

        # track_event(timestamp, entity, "stop", exp_dir, self._logger, detail=detail)
        track_event(
            timestamp,
            entity.name,
            entity.job_id,
            entity.step_id,
            entity.type,
            "stop",
            exp_dir,
            self._logger,
            detail=detail,
        )

    def on_timestep(self, timestamp: int, exp_dir: pathlib.Path) -> None:
        """Called at polling frequency to request status updates on
        monitored entities"""
        launcher = self.launcher
        entity_map = self._tracked_jobs

        names = {entity.name: entity for entity in entity_map.values()}

        if launcher and names:
            step_updates = launcher.get_step_update(list(names.keys()))

            for step_name, step_info in step_updates:
                if step_info and step_info.status in TERMINAL_STATUSES:
                    completed_entity = names[step_name]
                    self._to_completed(timestamp, completed_entity, exp_dir, step_info)


def event_loop(
    observer: BaseObserver,
    action_handler: ManifestEventHandler,
    frequency: t.Union[int, float],
    experiment_dir: pathlib.Path,
    num_iters: int,
    logger: logging.Logger,
) -> None:
    num_iters = num_iters if num_iters > 0 else 0  # ensure non-negative limits
    remaining = num_iters if num_iters else 0  # track completed iterations

    while observer.is_alive():
        timestamp = get_ts()
        logger.debug(f"Telemetry timestep: {timestamp}")
        action_handler.on_timestep(timestamp, experiment_dir)
        time.sleep(frequency)

        remaining -= 1
        if num_iters and not remaining:
            break


def main(
    frequency: t.Union[int, float],
    experiment_dir: pathlib.Path,
    logger: logging.Logger,
    observer: t.Optional[BaseObserver] = None,
    num_iters: int = 0,
) -> int:
    """Setup the monitoring entities and start the timer-based loop that
    will poll for telemetry data"""
    logger.info(
        f"Executing telemetry monitor with frequency: {frequency}"
        f", on target directory: {experiment_dir}"
    )

    manifest_path = experiment_dir / "manifest" / "manifest.json"
    manifest_dir = str(experiment_dir / "manifest")
    logger.debug(f"Monitoring manifest changes at: {manifest_dir}")

    log_handler = LoggingEventHandler(logger)  # type: ignore
    action_handler = ManifestEventHandler("manifest.json", logger)

    if observer is None:
        # create a file-system observer if one isn't injected
        observer = Observer()

    try:
        if manifest_path.exists():
            action_handler.process_manifest(str(manifest_path))

        observer.schedule(log_handler, manifest_dir)  # type: ignore
        observer.schedule(action_handler, manifest_dir)  # type: ignore
        observer.start()  # type: ignore

        event_loop(
            observer, action_handler, frequency, experiment_dir, num_iters, logger
        )
        return 0
    except Exception as ex:
        logger.error(ex)
    finally:
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
        "-f",
        type=str,
        help="Frequency of telemetry updates",
        required=True,
    )
    arg_parser.add_argument(
        "-d",
        type=str,
        help="Experiment root directory",
        required=True,
        # default="/lus/cls01029/mcbridch/ss/smartsim",
    )
    arg_parser.add_argument(
        "-n",
        type=int,
        help="Automatically shutdown after a specific number of polling iterations",
        default=0,
        required=False,
    )
    return arg_parser


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig()
    log = logging.getLogger()

    # Must register cleanup before the main loop is running
    register_signal_handlers()

    try:
        main(int(args.f), pathlib.Path(args.d), log)
        sys.exit(0)
    except Exception:
        log.exception(
            "Shutting down telemetry monitor due to unexpected error", exc_info=True
        )

    sys.exit(1)
