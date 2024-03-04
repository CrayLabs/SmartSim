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
import asyncio
import collections
import itertools
import logging
import pathlib
import typing as t

import redis.asyncio as redisa
import redis.exceptions as redisex
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from smartsim._core.control.job import JobEntity
from smartsim._core.utils.helpers import get_ts_ms

logger = logging.getLogger("TelemetryMonitor")


class Sink(abc.ABC):
    """Base class for telemetry output sinks"""

    @abc.abstractmethod
    async def save(self, *args: t.Any) -> None:
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
        filename = self._check_init(filename)
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

    async def save(self, *args: t.Any) -> None:
        """Save all arguments to a file as specified by the associated JobEntity"""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "a+", encoding="utf-8") as sink_fp:
            values = ",".join(map(str, args)) + "\n"
            sink_fp.write(values)


class Collector(abc.ABC):
    """Base class for metrics collectors"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the collector

        :param entity: The entity to collect metrics on
        :type entity: JobEntity"""
        self._entity = entity
        self._sink = sink
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entity(self) -> JobEntity:
        return self._entity

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
        return get_ts_ms() // 1000

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Execute any cleanup of resources for the collector"""


def find_collectors(entity: JobEntity) -> t.List[Collector]:
    """Map from manifest configuration to a set of possible collectors."""
    collectors: t.List[Collector] = []

    if entity.is_db and entity.telemetry_on:
        if mem_out := entity.collectors.get("memory", None):
            collectors.append(DBMemoryCollector(entity, FileSink(mem_out)))

        if con_out := entity.collectors.get("client", None):
            collectors.append(DBConnectionCollector(entity, FileSink(con_out)))

        if num_out := entity.collectors.get("client_count", None):
            collectors.append(DBConnectionCountCollector(entity, FileSink(num_out)))
    else:
        logger.debug(f"Collectors disabled for db {entity.name}")

    return collectors


class TaskStatusHandler(PatternMatchingEventHandler):
    """A file listener that will notify a set of collectors when an
    unmanaged entity has completed"""

    def __init__(
        self,
        collector: Collector,
        patterns: t.Optional[t.List[str]] = None,
        ignore_patterns: t.Optional[t.List[str]] = None,
        ignore_directories: bool = False,
        case_sensitive: bool = False,
    ):
        self._collectors = [collector]
        super().__init__(
            patterns, ignore_patterns, ignore_directories, case_sensitive
        )  # type: ignore

    def add(self, collector: Collector) -> None:
        self._collectors.append(collector)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Event handler for when a file or directory is modified.

        :param event: Event representing file/directory modification.
        :type event: FileModifiedEvent
        """
        super().on_modified(event)  # type: ignore
        self._notify(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event: Event representing file/directory creation.
        :type event: FileCreatedEvent
        """
        super().on_created(event)  # type: ignore
        self._notify(event.src_path)

    @abc.abstractmethod
    def _notify(self, event_src: str) -> None:
        """Notify the collector that the entity has stopped"""


class TaskCompleteHandler(TaskStatusHandler):
    def _notify(self, event_src: str) -> None:
        """Notify the collector that the entity has stopped"""
        logger.debug(f"Processing stop event created @ {event_src}")

        working_set = [item for item in self._collectors if item.enabled]
        for col in working_set:
            logger.debug(f"Disabling {col.entity.name}::{type(col).__name__}")
            col.disable()
            col.entity.set_complete()


class _Address:
    """Helper class to hold and pretty-print connection details"""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the instance"""
        self.host = host.strip() if host else ""
        self.port = port
        self._check()

    def _check(self) -> None:
        """Validate input arguments"""
        if not self.host:
            raise ValueError("Address requires host")
        if not self.port:
            raise ValueError("Address requires port")

    def __str__(self) -> str:
        """Pretty-print the instance"""
        return f"{self.host}:{self.port}"


class DBCollector(Collector):
    """A base class for collectors that retrieve statistics from an orchestrator"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the collector"""
        super().__init__(entity, sink)
        self._client: t.Optional[redisa.Redis[bytes]] = None
        self._address = _Address(
            self._entity.config.get("host", ""),
            int(self._entity.config.get("port", 0)),
        )

    async def _configure_client(self) -> None:
        """Configure and connect to the target database"""
        try:
            if not self._client:
                self._client = redisa.Redis(
                    host=self._address.host, port=self._address.port
                )
        except Exception as e:
            logger.exception(e)
        finally:
            if not self._client:
                logger.error(
                    f"{type(self).__name__} failed to connect to {self._address}"
                )

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
        try:
            if self._client:
                logger.info(
                    f"Shutting down {self._entity.name}::{self.__class__.__name__}"
                )
                await self._client.close()
                self._client = None
        except Exception as ex:
            logger.error(
                f"An error occurred during {type(self).__name__} shutdown", exc_info=ex
            )

    async def _check_db(self) -> bool:
        """Check if a database is reachable.

        :returns: True if connection succeeds, False otherwise."""
        try:
            if self._client:
                return await self._client.ping()
        except redisex.ConnectionError:
            logger.info(f"Cannot ping db {self._address}")

        return False


class DBMemoryCollector(DBCollector):
    """A collector that collects memory usage information from
    an orchestrator instance"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["used_memory", "used_memory_peak", "total_system_memory"]

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def collect(self) -> None:
        if not self.enabled:
            logger.debug(f"{type(self).__name__} is not enabled")
            return

        await self.prepare()
        if not self._client:
            logger.warning(f"{type(self).__name__} cannot collect")
            return

        try:
            if not await self._check_db():
                return

            db_info = await self._client.info("memory")
            metrics = (float(db_info[col]) for col in self._columns)
            value = (self.timestamp(), *metrics)

            await self._sink.save(*value)
        except Exception as ex:
            logger.warning(f"Collect failed for {type(self).__name__}", exc_info=ex)


class DBConnectionCollector(DBCollector):
    """A collector that collects client connection information from
    an orchestrator instance"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["client_id", "address"]

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def collect(self) -> None:
        if not self.enabled:
            logger.debug(f"{type(self).__name__} is not enabled")
            return

        await self.prepare()
        if not self._client:
            logger.warning(f"{type(self).__name__} is not connected and cannot collect")
            return

        now_ts = self.timestamp()  # ensure all results have the same timestamp

        try:
            if not await self._check_db():
                return

            clients = await self._client.client_list()

            all_metrics = ((now_ts, item["id"], item["addr"]) for item in clients)

            for metrics in all_metrics:
                await self._sink.save(*metrics)
        except Exception as ex:
            logger.warning(f"Collect failed for {type(self).__name__}", exc_info=ex)


class DBConnectionCountCollector(DBCollector):
    """A collector that collects client connection information from
    an orchestrator instance and records aggregated counts"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["num_clients"]

    async def post_prepare(self) -> None:
        """Hook called after the db connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def collect(self) -> None:
        if not self.enabled:
            logger.debug(f"{type(self).__name__} is not enabled")
            return

        await self.prepare()
        if not self._client:
            logger.warning(f"{type(self).__name__} is not connected and cannot collect")
            return

        try:
            if not await self._check_db():
                return

            client_list = await self._client.client_list()

            now_ts = self.timestamp()  # ensure all results have the same timestamp
            addresses = {item["addr"] for item in client_list}
            value = str(len(addresses))

            await self._sink.save(now_ts, value)
        except Exception as ex:
            logger.warning(f"Collect failed for {type(self).__name__}", exc_info=ex)


class CollectorManager:
    def __init__(self, timeout_ms: int = 1000) -> None:
        """Initialize the collector manager with an empty set of collectors
        :param timeout_ms: Timout (in ms) for telemetry collection
        :type timeout_ms: int
        """
        self._collectors: t.Dict[str, t.List[Collector]] = collections.defaultdict(list)
        self._timeout_ms = timeout_ms
        self._tasks: t.List[asyncio.Task[None]] = []
        self._stoppers: t.Dict[str, t.List[TaskCompleteHandler]] = (
            collections.defaultdict(list)
        )
        self._observers: t.Dict[str, t.List[BaseObserver]] = collections.defaultdict(
            list
        )

    def clear(self) -> None:
        """Remove all collectors from the managed set"""
        self._collectors = collections.defaultdict(list)

    def add(self, col: Collector) -> None:
        """Add a new collector to the managed set"""
        registered_collectors = self._collectors[col.owner]

        # ensure we only have 1 instance of a collector type registered
        if any(x for x in registered_collectors if type(x) is type(col)):
            return

        logger.debug(f"Adding collector: {col.owner}::{type(col).__name__}")
        self._collectors[col.owner].append(col)
        self._create_stop_listener(col)

    def _create_stop_listener(self, collector: Collector) -> None:
        """Create a file system listener to trigger a shutdown callback
        when an entity has completed processing"""
        if self._stoppers[collector.owner]:
            # listener already registered, add collector to it
            stopper = self._stoppers[collector.owner][0]
            stopper.add(collector)
            return

        stopper = TaskCompleteHandler(collector, patterns=["stop.json"])

        # if dir DNE, the observer may fail to start correctly.
        entity_dir = pathlib.Path(collector.entity.status_dir)
        entity_dir.mkdir(parents=True, exist_ok=True)

        observer = Observer()
        observer.schedule(stopper, collector.entity.status_dir)  # type: ignore
        observer.start()  # type: ignore

        self._stoppers[collector.owner].append(stopper)
        self._observers[collector.owner].append(observer)

    def add_all(self, clist: t.Iterable[Collector]) -> None:
        """Add multiple collectors to the managed set"""
        for col in clist:
            self.add(col)

    async def remove_all(self, entities: t.Iterable[JobEntity]) -> None:
        """Remove all collectors for the supplied entities"""
        if not entities:
            return

        await asyncio.gather(*(self.remove(entity) for entity in entities))

    async def remove(self, entity: JobEntity) -> None:
        """Remove all collectors for the supplied entity"""
        registered = self._collectors.pop(entity.name, [])
        self._stoppers.pop(entity.name, [])
        observers = self._observers.pop(entity.name, [])

        if not registered:
            return

        logger.debug(f"Removing collectors registered for {entity.name}")

        asyncio.gather(*(col.shutdown() for col in registered))

        for observer in itertools.chain(observers):
            observer.stop()  # type: ignore
            observer.join()

    async def prepare(self) -> None:
        """Ensure all managed collectors have prepared for collection"""

        await asyncio.gather(*(col.prepare() for col in self.all_collectors))

    async def collect(self) -> None:
        """Execute collection for all managed collectors"""
        if collectors := self.all_collectors:
            if self._tasks:
                tasks = self._tasks  # tasks still in progress
            else:
                tasks = [asyncio.create_task(item.collect()) for item in collectors]

            _, pending = await asyncio.wait(tasks, timeout=self._timeout_ms / 1000.0)
            if pending:
                for remaining_task in pending:
                    remaining_task.cancel()
                logger.debug(f"Execution of {len(pending)} collectors timed out.")

            self._tasks = []

    async def shutdown(self) -> None:
        """Release resources"""
        logger.debug(f"{type(self).__name__} cancelling tasks...")
        for task in self._tasks:
            if not task.done():
                task.cancel()

        logger.debug(f"{type(self).__name__} shutting down collectors...")
        if list(self.all_collectors):
            shutdown_tasks = [
                asyncio.create_task(item.shutdown()) for item in self.all_collectors
            ]
            await asyncio.wait(shutdown_tasks)
        logger.debug("Collector shutdown complete...")

    @property
    def all_collectors(self) -> t.Sequence[Collector]:
        """Get a list of all managed collectors"""
        collectors = itertools.chain.from_iterable(self._collectors.values())
        return [col for col in collectors if col.enabled]

    @property
    def dead_collectors(self) -> t.Sequence[Collector]:
        collectors = itertools.chain.from_iterable(self._collectors.values())
        return [col for col in collectors if not col.enabled]
