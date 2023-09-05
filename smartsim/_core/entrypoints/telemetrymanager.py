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
from dataclasses import dataclass, field
import json
import os
import pathlib
import signal
import typing as t
import asyncio


from types import FrameType

import logging
from datetime import datetime


from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, LoggingEventHandler
from watchdog.events import FileCreatedEvent, FileModifiedEvent

# from smartsim._core.launcher.launcher import Launcher
# from smartsim.wlm import detect_launcher

logging.basicConfig(level=logging.INFO)

"""
Telemetry Monitor entrypoint
"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
_EventClass = t.Literal["start", "stop", "timestep"]
_ManifestKey = t.Literal["timestamp", "applications", "orchestrators", "ensembles"]
_JobKey = t.Tuple[str, str]


@dataclass
class PersistableEntity:
    entity_type: str
    name: str
    job_id: str
    step_id: str

    @property
    def key(self) -> _JobKey:
        return (self.job_id, self.step_id)


_FilterFn = t.Callable[[PersistableEntity], bool]


@dataclass
class EntityEvent(PersistableEntity):
    timestamp: int


@dataclass
class Run:
    timestamp: int
    applications: t.List[PersistableEntity]
    orchestrators: t.List[PersistableEntity]
    ensembles: t.List[PersistableEntity]

    def flatten(
        self, filter_fn: t.Optional[_FilterFn] = None
    ) -> t.List[PersistableEntity]:
        """Flatten runs into a list of SmartSimEntity run events"""
        entities = self.applications + self.orchestrators + self.ensembles
        if filter_fn:
            entities = [entity for entity in entities if filter_fn(entity)]
        return entities


@dataclass
class ExperimentManifest:
    name: str
    path: pathlib.Path
    launcher: str
    runs: t.List[Run] = field(default_factory=list)


def hydrate_persistable(
    entity_type: str, persisted_entity: t.Dict[str, t.Any]
) -> PersistableEntity:
    return PersistableEntity(
        entity_type=entity_type,
        name=persisted_entity["name"],
        job_id=persisted_entity.get("job_id", ""),
        step_id=persisted_entity.get("step_id", ""),
    )


def hydrate_persistables(
    entity_type: _ManifestKey, run: t.Dict[_ManifestKey, t.Any]
) -> t.List[PersistableEntity]:
    return [hydrate_persistable(entity_type, item) for item in run[entity_type]]


def hydrate_runs(persisted_runs: t.List[t.Dict[_ManifestKey, t.Any]]) -> t.List[Run]:
    runs = [
        Run(
            timestamp=instance["timestamp"],
            applications=hydrate_persistables("applications", instance),
            orchestrators=hydrate_persistables("orchestrators", instance),
            ensembles=hydrate_persistables("ensembles", instance),
        )
        for instance in persisted_runs
    ]
    return runs


def load_manifest(file_path: str) -> ExperimentManifest:
    """Load a persisted manifest and return the content"""
    source = pathlib.Path(file_path)
    text = source.read_text(encoding="utf-8")
    manifest_dict = json.loads(text)

    manifest = ExperimentManifest(
        name=manifest_dict["experiment"]["name"],
        path=manifest_dict["experiment"]["path"],
        launcher=manifest_dict["experiment"]["launcher"],
        runs=hydrate_runs(manifest_dict["runs"]),
    )
    return manifest


def track_event(
    run: Run, entity: PersistableEntity, action: _EventClass, exp_dir: pathlib.Path
) -> None:
    """
    Persist a tracking event for an entity
    """
    job_id = entity.job_id or "missing_job_id"
    step_id = entity.step_id or "missing_step_id"
    entity_type = entity.entity_type or "missing_entity_type"
    print(
        f"mocked tracking {entity_type} event w/jid: {job_id}, "
        f"tid: {step_id}, ts: {run.timestamp}"
    )

    name: str = entity.name or "entity-name-not-found"
    tgt_path = exp_dir / "manifest" / entity_type / name / f"{action}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    persist = EntityEvent(
        entity.entity_type, entity.name, entity.job_id, entity.step_id, run.timestamp
    )
    tgt_path.write_text(json.dumps(persist))


class ManifestEventHandler(PatternMatchingEventHandler):
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
        self._tracked_jobs: t.Dict[_JobKey, PersistableEntity] = {}
        self._completed_jobs: t.Dict[_JobKey, PersistableEntity] = {}
        self._launcher: str = ""

    @property
    def tracked_runs(self) -> t.Dict[int, Run]:
        return self._tracked_runs

    @property
    def tracked_jobs(self) -> t.Dict[_JobKey, PersistableEntity]:
        return self._tracked_jobs

    @property
    def completed_jobs(self) -> t.Dict[_JobKey, PersistableEntity]:
        return self._completed_jobs

    @property
    def launcher(self) -> str:
        return self._launcher

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Called when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """
        super().on_modified(event)  # type: ignore

        # load items to process from manifest
        manifest = load_manifest(event.src_path)
        self._launcher = manifest.launcher

        runs = [run for run in manifest.runs if run.timestamp not in self.tracked_runs]

        # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        exp_dir = pathlib.Path(event.src_path).parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs
            ):
                # if entity.key not in self._tracked_jobs:
                self._tracked_jobs[entity.key] = entity
                track_event(run, entity, "start", exp_dir)
            self._tracked_runs[run.timestamp] = run

    def on_created(self, event: FileCreatedEvent) -> None:
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
        super().on_created(event)  # type: ignore

        # # load items to process from manifest
        # manifest = load_manifest(event.src_path)
        # self._launcher = manifest.launcher
        # runs = [run for run in manifest.runs if run.timestamp not in self.tracked_runs]

        # # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        # exp_dir = pathlib.Path(event.src_path).parent.parent

        # for run in runs:
        #     for entity in run.flatten(
        #         filter_fn=lambda e: e.key not in self._tracked_jobs
        #     ):
        #         if entity.key not in self._tracked_jobs:
        #             self._tracked_jobs[entity.key] = entity
        #             track_event(run, entity, "start", exp_dir)
        #     self._tracked_runs[run.timestamp] = run


def on_timestep(action_handler: ManifestEventHandler) -> None:
    # todo: update the completed jobs set in the manifest event handler when req'd
    # entity_names = {entity.name for entity in action_handler.tracked_jobs.values()}

    # launcher = None
    if action_handler.launcher in ["local", "slurm"]:
        # launcher = detect_launcher()
        # launcher = None
        ...

    # if launcher and entity_names:
    #     entity_statuses = launcher.get_step_update(entity_names)
    #     print(entity_statuses)


async def main(
    frequency: t.Union[int, float], experiment_dir: pathlib.Path, logger: logging.Logger
) -> None:
    logger.info(
        f"Executing telemetry monitor with frequency: {frequency}"
        f", on target directory: {experiment_dir}"
    )

    manifest_dir = str(experiment_dir / "manifest")
    logger.debug(f"Monitoring manifest changes at: {manifest_dir}")

    log_handler = LoggingEventHandler(logger)  # type: ignore
    action_handler = ManifestEventHandler("manifest.json", logger)

    observer = Observer()

    try:
        observer.schedule(log_handler, manifest_dir)  # type: ignore
        observer.schedule(action_handler, manifest_dir)  # type: ignore
        observer.start()  # type: ignore

        while observer.is_alive():
            logger.debug(f"Telemetry timestep: {datetime.timestamp(datetime.now())}")
            on_timestep(action_handler)
            await asyncio.sleep(frequency)
    except Exception as ex:
        logger.error(ex)
    finally:
        observer.stop()  # type: ignore
        observer.join()


if __name__ == "__main__":

    def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
        if not signo:
            logger = logging.getLogger()
            logger.warning("Received signal with no signo")

        loop = asyncio.get_event_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(description="SmartSim Telemetry Monitor")
    parser.add_argument(
        "-f", type=str, help="Frequency of telemetry updates", default=5
    )
    parser.add_argument(
        "-d",
        type=str,
        help="Experiment root directory",
        # required=True,
        default="/Users/chris.mcbride/code/ss/smartsim/_core/entrypoints",
    )

    args = parser.parse_args()

    # Register the cleanup before the main loop is running
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)

    try:
        asyncio.run(main(int(args.f), pathlib.Path(args.d), logging.getLogger()))
    except asyncio.CancelledError:
        print("Shutting down telemetry monitor...")
