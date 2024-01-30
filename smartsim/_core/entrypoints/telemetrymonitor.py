# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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
import abc
import argparse
import asyncio
import collections
import dataclasses
import datetime
import itertools
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

import redis.asyncio as redis
from anyio import open_file, sleep
from watchdog.events import (
    FileCreatedEvent,
    FileModifiedEvent,
    LoggingEventHandler,
    PatternMatchingEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from smartsim._core.config import CONFIG
from smartsim._core.control.job import JobEntity, _JobKey
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.launcher.launcher import Launcher
from smartsim._core.launcher.local.local import LocalLauncher
from smartsim._core.launcher.lsf.lsfLauncher import LSFLauncher
from smartsim._core.launcher.pbs.pbsLauncher import PBSLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher
from smartsim._core.launcher.stepInfo import StepInfo
from smartsim._core.utils.helpers import get_ts
from smartsim._core.utils.serialize import MANIFEST_FILENAME
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger
from smartsim.status import STATUS_COMPLETED, TERMINAL_STATUSES

# pylint: disable=too-many-lines
"""Telemetry Monitor entrypoint"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
_EventClass = t.Literal["start", "stop", "timestep"]
_MAX_MANIFEST_LOAD_ATTEMPTS: t.Final[int] = 6
F_MIN, F_MAX = 1.0, 600.0

logger = get_logger(__name__)


class Sink(abc.ABC):
    """Base class for telemetry output sinks"""

    @abc.abstractmethod
    async def save(self, **kwargs: t.Any) -> None:
        """Save the passed data to the underlying sink"""


class FileSink(Sink):
    """Telemetry sink that writes to a file"""

    def __init__(self, filename: str) -> None:
        """Initialize the FileSink
        :param entity: The JobEntity producing log data
        :type entity: JobEntity
        :param filename: The relative path and filename of the file to be written
        :type filename: str"""
        super().__init__()
        filename = FileSink._check_init(filename)
        self._path = pathlib.Path(filename)

    @staticmethod
    def _check_init(filename: str) -> str:
        """Validate initialization arguments.
        raise ValueError if an invalid filename is passed"""

        if not filename:
            raise ValueError("No filename provided to FileSink")

        return filename

    @property
    def path(self) -> pathlib.Path:
        """Returns the path to the underlying file the FileSink will write to"""
        return self._path

    async def save(self, **kwargs: t.Any) -> None:
        """Save all arguments to a file as specified by the associated JobEntity"""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        async with await open_file(self._path, "a+", encoding="utf-8") as sink_fp:
            values = ",".join(list(map(str, kwargs.values()))) + "\n"
            await sink_fp.write(values)


class Collector(abc.ABC):
    """Base class for metrics collectors"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the collector

        :param entity: The entity to collect metrics on
        :type entity: JobEntity"""
        self._entity = entity
        self._sink = sink
        self._value: t.Any = None

    @property
    def owner(self) -> str:
        """The name of the SmartSim entity the collector is attached to"""
        return self._entity.name

    @property
    def sink(self) -> Sink:
        """The sink where collected data is written"""
        return self._sink

    @abc.abstractmethod
    async def prepare(self) -> None:
        """Initialization logic for a collector"""

    @abc.abstractmethod
    async def collect(self) -> None:
        """Execute metric collection against a producer"""

    @staticmethod
    def timestamp() -> int:
        """Return an integer timestamp"""
        return int(datetime.datetime.timestamp(datetime.datetime.now()))

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Execute any cleanup of resources for the collector"""


@dataclasses.dataclass
class _Address:
    """Helper class to hold and pretty-print connection details"""

    host: str
    port: int

    def __init__(self, host: str, port: int) -> None:
        """Initialize the instance"""
        self.host = host.strip() if host else ""
        self.port = port
        _Address._check(self.host, self.port)

    @staticmethod
    def _check(host: str, port: int) -> None:
        """Validate input arguments"""
        if not host:
            raise ValueError("Address requires host")
        if not port:
            raise ValueError("Address requires port")

    def __str__(self) -> str:
        """Pretty-print the instance"""
        return f"{self.host}:{self.port}"


class DbCollector(Collector):
    """A base class for collectors that retrieve statistics from an orchestrator"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the collector"""
        super().__init__(entity, sink)
        self._client: t.Optional[redis.Redis[bytes]] = None
        self._address = _Address(
            self._entity.config.get("host", ""),
            int(self._entity.config.get("port", 0)),
        )

    async def _configure_client(self) -> None:
        """Configure and connect to the target database"""
        try:
            if not self._client:
                self._client = redis.Redis(
                    host=self._address.host, port=self._address.port
                )
        except ConnectionRefusedError as cre:
            logger.exception(cre)
        except Exception as e:
            logger.exception(e)
        finally:
            if not self._client:
                logger.error(f"DbCollector failed to connect to {self._address}")

    async def prepare(self) -> None:
        """Initialization logic for a DB collector"""
        if self._client:
            return

        await self._configure_client()
        await self.post_prepare()

    @abc.abstractmethod
    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""

    async def shutdown(self) -> None:
        """Release any resources held by the collector"""
        logger.debug(f"Shutting down DbCollector for {self._entity.name}")

        try:
            if self._client:
                await self._client.close()
                self._client = None
        except Exception as ex:
            logger.error("An error occurred during DbCollector shutdown", exc_info=ex)


class DbMemoryCollector(DbCollector):
    """A collector that collects memory usage information from
    an orchestrator instance"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["used_memory", "used_memory_peak", "total_system_memory"]

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save(
            col0="timestamp",
            col1=self._columns[0],
            col2=self._columns[1],
            col3=self._columns[2],
        )

    async def collect(self) -> None:
        await self.prepare()
        if not self._client:
            logger.warning("DbMemoryCollector cannot collect")
            return

        self._value = {}

        try:
            db_info = await self._client.info("memory")
            for key in self._columns:
                self._value[key] = db_info[key]

            await self._sink.save(timestamp=self.timestamp(), **self._value)
        except Exception as ex:
            logger.warning("collect failed for DbMemoryCollector", exc_info=ex)

    @property
    def value(self) -> t.Dict[str, int]:
        filtered: t.Dict[str, int] = self._value
        self._value = None
        return filtered


class DbConnectionCollector(DbCollector):
    """A collector that collects client connection information from
    an orchestrator instance"""

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save(col0="timestamp", col1="client_id", col2="address")

    async def collect(self) -> None:
        await self.prepare()
        if not self._client:
            logger.warning("DbConnectionCollector is not connected and cannot collect")
            return

        now_ts = self.timestamp()  # ensure all results have the same timestamp

        try:
            clients = await self._client.client_list()

            self._value = [{"addr": item["addr"], "id": item["id"]} for item in clients]

            for client_info in self._value:
                _id, _ip = client_info["id"], client_info["addr"]
                await self._sink.save(timestamp=now_ts, client_id=_id, address=_ip)
        except Exception as ex:
            logger.warning("collect failed for DbMemoryCollector", exc_info=ex)

    @property
    def value(self) -> t.List[str]:
        filtered: t.List[str] = self._value
        self._value = None
        return filtered


class DbConnectionCountCollector(DbCollector):
    """A collector that collects client connection information from
    an orchestrator instance and records aggregated counts"""

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save(col0="timestamp", col1="num_clients")

    async def collect(self) -> None:
        await self.prepare()
        if not self._client:
            logger.warning(
                "DbConnectionCountCollector is not connected and cannot collect"
            )
            return

        try:
            client_list = await self._client.client_list()

            now_ts = self.timestamp()  # ensure all results have the same timestamp
            addresses = {item["addr"] for item in client_list}
            self._value = str(len(addresses))

            await self._sink.save(timestamp=now_ts, num_clients=self._value)
        except Exception as ex:
            logger.warning("collect failed for DbConnectionCountCollector", exc_info=ex)

    @property
    def value(self) -> str:
        filtered = str(self._value)
        self._value = None
        return filtered


class CollectorManager:
    def __init__(self, timeout_ms: int = 1000) -> None:
        """Initialize the collector manager with an empty set of collectors
        :param timeout_ms: Timout (in ms) for telemetry collection
        :type timeout_ms: int
        """
        self._collectors: t.Dict[str, t.List[Collector]] = collections.defaultdict(
            lambda: []
        )
        self._timeout_ms = timeout_ms
        self._tasks: t.List[asyncio.Task[None]] = []

    def clear(self) -> None:
        """Remove all collectors from the managed set"""
        self._collectors = collections.defaultdict(lambda: [])

    def add(self, col: Collector) -> None:
        """Add a new collector to the managed set"""
        self.add_all([col])

    def add_all(self, clist: t.Iterable[Collector]) -> None:
        """Add multiple collectors to the managed set"""
        for col in clist:
            owner_list = self._collectors[col.owner]
            dupes = next((x for x in owner_list if type(x) is type(col)), None)
            if dupes:
                continue

            self._collectors[col.owner].append(col)

    def remove(self, entity: JobEntity) -> None:
        registered = self._collectors.pop(entity.name, None)
        if registered:
            # shutdown this set of collectors
            # todo: move shutdown logic into static so i can pass a collection instead
            # of always shutting down everything...
            logger.debug(f"removing collectors registered for {entity.name}")

    async def prepare(self) -> None:
        """Ensure all managed collectors have prepared for collection"""
        for collector in self.all_collectors:
            await collector.prepare()

    async def collect(self) -> None:
        """Execute collection for all managed collectors"""
        logger.debug("Executing all telemetry collectors")

        if collectors := self.all_collectors:
            if self._tasks:  # tasks still in progress
                return

            self._tasks = [asyncio.create_task(item.collect()) for item in collectors]
            results = await asyncio.wait(self._tasks, timeout=self._timeout_ms / 1000.0)
            logger.debug(f"collector.collect() results: {results}")
            self._tasks.clear()

    async def shutdown(self) -> None:
        """Release resources"""
        logger.debug("CollectorManager cancelling tasks...")
        for task in self._tasks:
            if not task.done():
                task.cancel()

        logger.debug("CollectorManager shutting down collectors...")
        if list(self.all_collectors):
            shutdown_tasks = [
                asyncio.create_task(item.shutdown()) for item in self.all_collectors
            ]
            await asyncio.wait(shutdown_tasks)
        logger.debug("Collector shutdown complete...")

    @classmethod
    def find_collectors(cls, entity: JobEntity) -> t.List[Collector]:
        """Map from manifest configuration to a set of possible collectors."""
        collectors: t.List[Collector] = []

        if entity.is_db and entity.telemetry_on:
            if mem_out := entity.collectors.get("memory", None):
                collectors.append(DbMemoryCollector(entity, FileSink(mem_out)))

            if con_out := entity.collectors.get("client", None):
                collectors.append(DbConnectionCollector(entity, FileSink(con_out)))

            if num_out := entity.collectors.get("client_count", None):
                collectors.append(DbConnectionCountCollector(entity, FileSink(num_out)))
        else:
            logger.debug(f"collectors disabled for db {entity.name}")

        return collectors

    @property
    def all_collectors(self) -> t.Iterable[Collector]:
        """Get a list of all managed collectors"""
        return list(itertools.chain(*self._collectors.values()))


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
    at runtime to satisfy the experiment requirements.
    """

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

    if entity.is_db:
        # db shards are hydrated individually
        entity.telemetry_on = int(persistable_entity.get("collectors", "0")) > 0
        col_cfg = entity.collectors

        if entity.telemetry_on:
            col_cfg["client"] = persistable_entity.get("client_file", "")
            col_cfg["client_count"] = persistable_entity.get("client_count_file", "")
            col_cfg["memory"] = persistable_entity.get("memory_file", "")

        entity.config["host"] = persistable_entity.get("hostname", "")
        entity.config["port"] = persistable_entity.get("port", "")

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

    while manifest_dict is None and try_count < _MAX_MANIFEST_LOAD_ATTEMPTS:
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
    detail: str = "",
    return_code: t.Optional[int] = None,
) -> None:
    """Persist a tracking event for an entity"""
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


def faux_return_code(step_info: StepInfo) -> t.Optional[int]:
    """Create a faux return code for a task run by the WLM. Must not be
    called with non-terminal statuses or results may be confusing
    """
    rc_map = {s: 1 for s in TERMINAL_STATUSES}  # return `1` for all terminal statuses
    rc_map.update({STATUS_COMPLETED: os.EX_OK})  # return `0` for full success

    return rc_map.get(step_info.status, None)  # return `None` when in-progress


class ManifestEventHandler(PatternMatchingEventHandler):
    """The ManifestEventHandler monitors an experiment for changes and updates
    a telemetry datastore as needed.

    It contains event handlers that are triggered by changes to a runtime experiment
    manifest. The runtime manifest differs from a standard manifest. A runtime manifest
    may contain multiple experiment executions in a `runs` collection.

    It also contains a long-polling loop that checks experiment entities for updates
    at each timestep.
    """

    def __init__(
        self,
        pattern: str,
        ignore_patterns: t.Any = None,
        ignore_directories: bool = True,
        case_sensitive: bool = False,
        timeout_ms: int = 1000,
    ) -> None:
        super().__init__(
            [pattern], ignore_patterns, ignore_directories, case_sensitive
        )  # type: ignore
        self._tracked_runs: t.Dict[int, Run] = {}
        self._tracked_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._completed_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._launcher: t.Optional[Launcher] = None
        self.job_manager: JobManager = JobManager(threading.RLock())
        self._launcher_map: t.Dict[str, t.Type[Launcher]] = {
            "slurm": SlurmLauncher,
            "pbs": PBSLauncher,
            "lsf": LSFLauncher,
            "local": LocalLauncher,
        }
        self._timeout_ms = timeout_ms
        self._collector = CollectorManager(timeout_ms)

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    def init_launcher(self, launcher: str) -> Launcher:
        """Initialize the controller with a specific type of launcher.
        SmartSim currently supports slurm, pbs(pro), lsf,
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
        :type manifest_path: str
        """
        try:
            manifest = load_manifest(manifest_path)
            if not manifest:
                return
        except json.JSONDecodeError:
            logger.error(f"Malformed manifest encountered: {manifest_path}")
            return
        except ValueError:
            logger.error("Manifest content error", exc_info=True)
            return

        if self._launcher is None:
            self.set_launcher(manifest.launcher)

        if not self._launcher:
            raise SmartSimError(f"Unable to set launcher from {manifest_path}")

        runs = [run for run in manifest.runs if run.timestamp not in self._tracked_runs]
        exp_dir = pathlib.Path(manifest_path).parent.parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs
            ):
                entity.path = str(exp_dir)

                self._tracked_jobs[entity.key] = entity

                if entity.telemetry_on:
                    collectors = CollectorManager.find_collectors(entity)
                    self._collector.add_all(collectors)

                track_event(
                    run.timestamp,
                    entity.task_id,
                    entity.step_id,
                    entity.type,
                    "start",
                    pathlib.Path(entity.status_dir),
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
        :type event: FileModifiedEvent
        """
        super().on_modified(event)  # type: ignore
        logger.debug(f"processing manifest modified @ {event.src_path}")
        self.process_manifest(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event: Event representing file/directory creation.
        :type event: FileCreatedEvent
        """
        super().on_created(event)  # type: ignore
        logger.debug(f"processing manifest created @ {event.src_path}")
        self.process_manifest(event.src_path)

    def _to_completed(
        self,
        timestamp: int,
        entity: JobEntity,
        step_info: StepInfo,
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
        :type entity: StepInfo
        """
        inactive_entity = self._tracked_jobs.pop(entity.key)
        if entity.key not in self._completed_jobs:
            self._completed_jobs[entity.key] = inactive_entity

        job = self.job_manager[entity.name]
        self.job_manager.move_to_completed(job)

        status_clause = f"status: {step_info.status}"
        error_clause = f", error: {step_info.error}" if step_info.error else ""

        if hasattr(job.entity, "status_dir"):
            write_path = pathlib.Path(job.entity.status_dir)

        track_event(
            timestamp,
            entity.task_id,
            entity.step_id,
            entity.type,
            "stop",
            write_path,
            detail=f"{status_clause}{error_clause}",
            return_code=faux_return_code(step_info),
        )

        # NOTE: this doesn't work because databases aren't in the tracked jobs?
        # todo: make sure we have a tracking mechanism for stop events for unmanaged
        # items so we can unregister...
        self._collector.remove(entity)

    async def on_timestep(self, timestamp: int) -> None:
        """Called at polling frequency to request status updates on
        monitored entities

        :param timestamp: the current timestamp for event logging
        :type timestamp: int
        :param experiment_dir: the experiement directory to monitor for changes
        :type experiment_dir: pathlib.Path
        """
        if not self._launcher:
            return

        await self._collector.collect()

        # consider not using name to avoid collisions
        if names := {entity.name: entity for entity in self._tracked_jobs.values()}:
            step_updates = self._launcher.get_step_update(list(names.keys()))

            for step_name, step_info in step_updates:
                if step_info and step_info.status in TERMINAL_STATUSES:
                    completed_entity = names[step_name]
                    self._to_completed(timestamp, completed_entity, step_info)

    async def shutdown(self) -> None:
        logger.debug("ManifestEventHandler shutting down...")
        await self._collector.shutdown()
        logger.debug("ManifestEventHandler shutdown complete...")


def can_shutdown(action_handler: ManifestEventHandler) -> bool:
    jobs = action_handler.job_manager.jobs
    db_jobs = action_handler.job_manager.db_jobs

    if has_jobs := bool(jobs):
        logger.debug(f"telemetry monitor is monitoring {len(jobs)} jobs")
    if has_dbs := bool(db_jobs):
        logger.debug(f"telemetry monitor is monitoring {len(db_jobs)} dbs")

    has_running_jobs = has_jobs or has_dbs
    return not has_running_jobs


async def event_loop(
    observer: BaseObserver,
    action_handler: ManifestEventHandler,
    frequency: t.Union[int, float],
    cooldown_duration: int,
) -> None:
    """Executes all attached timestep handlers every <frequency> seconds

    :param observer: (optional) a preconfigured watchdog Observer to inject
    :type observer: t.Optional[BaseObserver]
    :param action_handler: The manifest event processor instance
    :type action_handler: ManifestEventHandler
    :param frequency: frequency (in seconds) of update loop
    :type frequency: t.Union[int, float]
    :param logger: a preconfigured Logger instance
    :type logger: logging.Logger
    :param cooldown_duration: number of seconds the telemetry monitor should
                              poll for new jobs before attempting to shutdown
    :type cooldown_duration: int
    """
    elapsed: int = 0
    last_ts: int = get_ts()
    shutdown_in_progress = False

    while observer.is_alive() and not shutdown_in_progress:
        duration_ms = 0
        start_ts = get_ts()
        logger.debug(f"Telemetry timestep: {start_ts}")
        await action_handler.on_timestep(start_ts)

        elapsed += start_ts - last_ts
        last_ts = start_ts

        if can_shutdown(action_handler):
            if elapsed >= cooldown_duration:
                shutdown_in_progress = True
                logger.debug("beginning telemetry manager shutdown")
                await action_handler.shutdown()
                logger.debug("beginning file monitor shutdown")
                observer.stop()  # type: ignore
                logger.debug("Event loop shutdown complete")
                break
        else:
            # reset cooldown any time there are still jobs running
            elapsed = 0

        # track time elapsed to execute metric collection
        duration_ms = get_ts() - start_ts
        wait_ms = max(frequency - duration_ms, 0)
        logger.debug(
            f"Collectors consumed {duration_ms} ms of {action_handler.timeout_ms}"
            f" ms loop frequency. Sleeping {wait_ms} ms",
        )

        # delay next loop if collection time didn't exceed loop frequency
        if wait_ms > 0:
            await sleep(wait_ms / 1000)  # convert to seconds for sleep

    logger.debug("Exiting telemetry monitor event loop")


async def main(
    frequency: t.Union[int, float],
    experiment_dir: pathlib.Path,
    observer: t.Optional[BaseObserver] = None,
    cooldown_duration: t.Optional[int] = 0,
) -> int:
    """Setup the monitoring entities and start the timer-based loop that
    will poll for telemetry data

    :param frequency: frequency (in seconds) of update loop
    :type frequency: t.Union[int, float]
    :param experiment_dir: the experiement directory to monitor for changes
    :type experiment_dir: pathlib.Path
    :param logger: a preconfigured Logger instance
    :type logger: logging.Logger
    :param observer: (optional) a preconfigured Observer to inject
    :type observer: t.Optional[BaseObserver]
    :param cooldown_duration: number of seconds the telemetry monitor should
                              poll for new jobs before attempting to shutdown
    :type cooldown_duration: int
    """
    manifest_relpath = pathlib.Path(CONFIG.telemetry_subdir) / MANIFEST_FILENAME
    manifest_path = experiment_dir / manifest_relpath
    monitor_pattern = str(manifest_relpath)

    logger.info(
        f"Executing telemetry monitor with frequency: {frequency}s"
        f", on target directory: {experiment_dir}"
        f" matching pattern: {monitor_pattern}"
    )

    cooldown_duration = cooldown_duration or CONFIG.telemetry_cooldown
    log_handler = LoggingEventHandler(logger)  # type: ignore
    frequency_ms = int(frequency * 1000)  # limit collector execution time
    action_handler = ManifestEventHandler(monitor_pattern, timeout_ms=frequency_ms)

    if observer is None:
        observer = Observer()

    try:
        if manifest_path.exists():
            # a manifest may not exist depending on startup timing
            action_handler.process_manifest(str(manifest_path))

        observer.schedule(log_handler, experiment_dir, recursive=True)  # type:ignore
        observer.schedule(action_handler, experiment_dir, recursive=True)  # type:ignore
        observer.start()  # type: ignore

        await event_loop(observer, action_handler, frequency_ms, cooldown_duration)
        return os.EX_OK
    except Exception as ex:
        logger.error(ex)
    finally:
        if observer.is_alive():
            observer.stop()  # type: ignore
            observer.join()
        await action_handler.shutdown()
        logger.debug("final telemetry monitor shutdown complete")

    return os.EX_SOFTWARE


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
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
        type=float,
        help="Frequency of telemetry updates (in seconds))",
        required=True,
    )
    arg_parser.add_argument(
        "-exp_dir",
        type=str,
        help="Experiment root directory",
        required=True,
    )
    arg_parser.add_argument(
        "-cooldown",
        type=int,
        help="Default lifetime of telemetry monitor (in seconds) before auto-shutdown",
        default=CONFIG.telemetry_cooldown,
    )
    return arg_parser


def check_frequency(frequency: t.Union[int, float]) -> None:
    freq_tpl = "Telemetry collection frequency must be {0} {1}s"
    if frequency < F_MIN:
        raise ValueError(freq_tpl.format("greater than", F_MIN))
    if frequency > F_MAX:
        raise ValueError(freq_tpl.format("less than", F_MAX))


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = get_parser()
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    log_path = os.path.join(
        args.exp_dir, CONFIG.telemetry_subdir, "telemetrymonitor.log"
    )
    fh = logging.FileHandler(log_path, "a")
    logger.addHandler(fh)

    # Must register cleanup before the main loop is running
    register_signal_handlers()
    check_frequency(float(args.frequency))

    try:
        asyncio.run(
            main(
                int(args.frequency),
                pathlib.Path(args.exp_dir),
                cooldown_duration=args.cooldown,
            )
        )
        sys.exit(0)
    except Exception:
        logger.exception(
            "Shutting down telemetry monitor due to unexpected error", exc_info=True
        )

    sys.exit(1)
