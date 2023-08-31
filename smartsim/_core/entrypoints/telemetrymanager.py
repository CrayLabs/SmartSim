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
import copy
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

logging.basicConfig(level=logging.INFO)

"""
Telemetry Monitor entrypoint
"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]


def load_manifest(file_path: str) -> t.Dict[str, t.Any]:
    source = pathlib.Path(file_path)
    text = source.read_text(encoding="utf-8")
    manifest = json.loads(text)
    return manifest


def filter_new(
    runs: t.List[t.Dict[str, t.Dict[str, t.Any]]], tracked: t.List[int]
) -> t.List[t.Dict[str, t.Dict[str, t.Any]]]:
    # filter out items that are already in the list of tracked runs
    result = [run for run in runs if run["timestamp"] not in tracked]
    return result


def reshape_run(
    runs: t.List[t.Dict[str, t.Dict[str, t.Any]]]
) -> t.Dict[int, t.Dict[str, t.Any]]:
    # remove the non-entity keys
    return {
        run["timestamp"]: {
            "applications": run["applications"],
            "orchestrators": run["orchestrators"],
            "ensembles": run["ensembles"],
        }
        for run in runs
    }


def track_event(
    timestamp: float, entity_type: str, entity: t.Dict[str, t.Any], action: str, exp_dir: pathlib.Path
) -> None:
    job_id = entity.get("job_id", "no job id")
    task_id = entity.get("step_id", "no task id")
    print(f"mocked tracking {entity_type} event w/jid: {job_id}, tid: {task_id}, ts: {timestamp}")

    name: str = entity.get("name", "entity-name-not-found")
    tgt_path = exp_dir / "manifest" / entity_type / name / f"{action}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    body = {
        "timestamp": timestamp,  # datetime.timestamp(datetime.now()),
        "job_id": job_id or "",
        "task_id": task_id or "",
    }
    tgt_path.write_text(json.dumps(body))


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
        # self._runs: t.Dict[int, t.Any] = {}
        # self._runs: t.List[int] = []
        self._tracked: t.List[int] = []

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
        raw_runs = manifest.get("runs", [])

        raw_runs = filter_new(raw_runs, self._tracked)
        runs = reshape_run(raw_runs)

        # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        exp_dir = pathlib.Path(event.src_path).parent.parent

        for ts, data in runs.items():
            for _type, entities in data.items():
                for entity in entities:
                    track_event(ts, _type, entity, "start", exp_dir)
            self._tracked.append(ts)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
        super().on_created(event)  # type: ignore

        # load items to process from manifest
        manifest = load_manifest(event.src_path)
        raw_runs = manifest.get("runs", [])

        raw_runs = filter_new(raw_runs, self._tracked)
        runs = reshape_run(raw_runs)

        # Find exp root assuming event path `{exp_root}/manifest/manifest.json`
        exp_dir = pathlib.Path(event.src_path).parent.parent

        for ts, data in runs.items():
            for _type, entities in data.items():
                for entity in entities:
                    track_event(ts, _type, entity, "start", exp_dir)
            self._tracked.append(ts)


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
