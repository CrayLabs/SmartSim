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
import typing as t

import redis.asyncio as redisa
import redis.exceptions as redisex

from smartsim._core.control.job import JobEntity
from smartsim._core.utils.helpers import get_ts_ms
from smartsim._core.utils.telemetry.sink import FileSink, Sink

logger = logging.getLogger("TelemetryMonitor")


class Collector(abc.ABC):
    """Base class for telemetry collectors.

    A Collector is used to retrieve runtime metrics about an entity."""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the collector

        :param entity: entity to collect metrics on
        :param sink: destination to write collected information
        """
        self._entity = entity
        self._sink = sink
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Boolean indicating if the collector should perform data collection"""
        return self._entity.telemetry_on

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._entity.telemetry_on = value

    @property
    def entity(self) -> JobEntity:
        """The `JobEntity` for which data is collected
        :return: the entity"""
        return self._entity

    @property
    def sink(self) -> Sink:
        """The sink where collected data is written
        :return: the sink
        """
        return self._sink

    @abc.abstractmethod
    async def prepare(self) -> None:
        """Initialization logic for the collector"""

    @abc.abstractmethod
    async def collect(self) -> None:
        """Execute metric collection"""

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Execute cleanup of resources for the collector"""


class _DBAddress:
    """Helper class to hold and pretty-print connection details"""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the instance
        :param host: host address for database connections
        :param port: port number for database connections
        """
        self.host = host.strip() if host else ""
        self.port = port
        self._check()

    def _check(self) -> None:
        """Validate input arguments"""
        if not self.host:
            raise ValueError(f"{type(self).__name__} requires host")
        if not self.port:
            raise ValueError(f"{type(self).__name__} requires port")

    def __str__(self) -> str:
        """Pretty-print the instance"""
        return f"{self.host}:{self.port}"


class DBCollector(Collector):
    """A base class for collectors that retrieve statistics from an orchestrator"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        """Initialize the `DBCollector`

        :param entity: entity with metadata about the resource to monitor
        :param sink: destination to write collected information
        """
        super().__init__(entity, sink)
        self._client: t.Optional[redisa.Redis[bytes]] = None
        self._address = _DBAddress(
            self._entity.config.get("host", ""),
            int(self._entity.config.get("port", 0)),
        )

    async def _configure_client(self) -> None:
        """Configure the client connection to the target database"""
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
        """Initialization logic for the DB collector. Creates a database
        connection then executes the `post_prepare` callback function."""
        if self._client:
            return

        await self._configure_client()
        await self._post_prepare()

    @abc.abstractmethod
    async def _post_prepare(self) -> None:
        """Hook function to enable subclasses to perform actions
        after a db client is ready"""

    @abc.abstractmethod
    async def _perform_collection(
        self,
    ) -> t.Sequence[t.Tuple[t.Union[int, float, str], ...]]:
        """Hook function for subclasses to execute custom metric retrieval.
        NOTE: all implementations return an iterable of metrics to avoid
        adding extraneous base class code to differentiate the results

        :return: an iterable containing individual metric collection results
        """

    async def collect(self) -> None:
        """Execute database metric collection if the collector is enabled. Writes
        the resulting metrics to the associated output sink. Calling `collect`
        when `self.enabled` is `False` performs no actions."""
        if not self.enabled:
            # collectors may be disabled by monitoring changes to the
            # manifest. Leave the collector but do NOT collect
            logger.debug(f"{type(self).__name__} is not enabled")
            return

        await self.prepare()
        if not self._client:
            logger.warning(f"{type(self).__name__} cannot collect")
            return

        try:
            # if we can't communicate w/the db, exit
            if not await self._check_db():
                return

            all_metrics = await self._perform_collection()
            for metrics in all_metrics:
                await self._sink.save(*metrics)
        except Exception as ex:
            logger.warning(f"Collect failed for {type(self).__name__}", exc_info=ex)

    async def shutdown(self) -> None:
        """Execute cleanup of database client connections"""
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
        """Check if the target database is reachable.

        :return: `True` if connection succeeds, `False` otherwise.
        """
        try:
            if self._client:
                return await self._client.ping()
        except redisex.ConnectionError:
            logger.warning(f"Cannot ping db {self._address}")

        return False


class DBMemoryCollector(DBCollector):
    """A `DBCollector` that collects memory consumption metrics"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["used_memory", "used_memory_peak", "total_system_memory"]

    async def _post_prepare(self) -> None:
        """Write column headers for a CSV formatted output sink after
        the database connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def _perform_collection(
        self,
    ) -> t.Sequence[t.Tuple[int, float, float, float]]:
        """Perform memory metric collection and return the results

        :return: an iterable containing individual metric collection results
        in the format `(timestamp,used_memory,used_memory_peak,total_system_memory)`
        """
        if self._client is None:
            return []

        db_info = await self._client.info("memory")

        used = float(db_info["used_memory"])
        peak = float(db_info["used_memory_peak"])
        total = float(db_info["total_system_memory"])

        value = (get_ts_ms(), used, peak, total)

        # return a list containing a single record to simplify the parent
        # class code to save multiple records from a single collection
        return [value]


class DBConnectionCollector(DBCollector):
    """A `DBCollector` that collects database client-connection metrics"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["client_id", "address"]

    async def _post_prepare(self) -> None:
        """Write column headers for a CSV formatted output sink after
        the database connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def _perform_collection(
        self,
    ) -> t.Sequence[t.Tuple[t.Union[int, str, str], ...]]:
        """Perform connection metric collection and return the results

        :return: an iterable containing individual metric collection results
        in the format `(timestamp,client_id,address)`
        """
        if self._client is None:
            return []

        now_ts = get_ts_ms()
        clients = await self._client.client_list()

        values: t.List[t.Tuple[int, str, str]] = []

        # content-filter the metrics and return them all together
        for client in clients:
            # all records for the request will have the same timestamp
            value = now_ts, client["id"], client["addr"]
            values.append(value)

        return values


class DBConnectionCountCollector(DBCollector):
    """A DBCollector that collects aggregated client-connection count metrics"""

    def __init__(self, entity: JobEntity, sink: Sink) -> None:
        super().__init__(entity, sink)
        self._columns = ["num_clients"]

    async def _post_prepare(self) -> None:
        """Write column headers for a CSV formatted output sink after
        the database connection is established"""
        await self._sink.save("timestamp", *self._columns)

    async def _perform_collection(
        self,
    ) -> t.Sequence[t.Tuple[int, int]]:
        """Perform connection-count metric collection and return the results

        :return: an iterable containing individual metric collection results
        in the format `(timestamp,num_clients)`
        """
        if self._client is None:
            return []

        client_list = await self._client.client_list()

        addresses = {item["addr"] for item in client_list}

        # return a list containing a single record to simplify the parent
        # class code to save multiple records from a single collection
        value = (get_ts_ms(), len(addresses))
        return [value]


class CollectorManager:
    """The `CollectorManager` manages the set of all collectors required to retrieve
    metrics for an experiment. It provides the ability to add and remove collectors
    with unique configuration per entity. The `CollectorManager` is primarily used
    to perform bulk actions on 1-to-many collectors (e.g. prepare all collectors,
    request metrics for all collectors, close all collector connections)"""

    def __init__(self, timeout_ms: int = 1000) -> None:
        """Initialize the `CollectorManager` without collectors
        :param timeout_ms: maximum time (in ms) allowed for `Collector.collect`
        """
        # A lookup table to hold a list of registered collectors per entity
        self._collectors: t.Dict[str, t.List[Collector]] = collections.defaultdict(list)
        # Max time to allow a collector to work before cancelling requests
        self._timeout_ms = timeout_ms

    def clear(self) -> None:
        """Remove all collectors from the monitored set"""
        self._collectors = collections.defaultdict(list)

    def add(self, collector: Collector) -> None:
        """Add a collector to the monitored set

        :param collector: `Collector` instance to monitor
        """
        entity_name = collector.entity.name

        registered_collectors = self._collectors[entity_name]

        # Exit if the collector is already registered to the entity
        if any(c for c in registered_collectors if type(c) is type(collector)):
            return

        logger.debug(f"Adding collector: {entity_name}::{type(collector).__name__}")
        registered_collectors.append(collector)

    def add_all(self, collectors: t.Sequence[Collector]) -> None:
        """Add multiple collectors to the monitored set

        :param collectors: a collection of `Collectors` to monitor
        """
        for collector in collectors:
            self.add(collector)

    async def remove_all(self, entities: t.Sequence[JobEntity]) -> None:
        """Remove all collectors registered to the supplied entities

        :param entities: a collection of `JobEntity` instances that will
        no longer have registered collectors
        """
        if not entities:
            return

        tasks = (self.remove(entity) for entity in entities)
        await asyncio.gather(*tasks)

    async def remove(self, entity: JobEntity) -> None:
        """Remove all collectors registered to the supplied entity

        :param entities: `JobEntity` that will no longer have registered collectors
        """
        registered = self._collectors.pop(entity.name, [])
        if not registered:
            return

        logger.debug(f"Removing collectors registered for {entity.name}")
        asyncio.gather(*(collector.shutdown() for collector in registered))

    async def prepare(self) -> None:
        """Prepare registered collectors to perform collection"""
        tasks = (collector.prepare() for collector in self.all_collectors)
        # use gather so all collectors are prepared before collection
        await asyncio.gather(*tasks)

    async def collect(self) -> None:
        """Perform collection for all registered collectors"""
        if collectors := self.all_collectors:
            tasks = [asyncio.create_task(item.collect()) for item in collectors]

            _, pending = await asyncio.wait(tasks, timeout=self._timeout_ms / 1000.0)

            # any tasks still pending has exceeded the timeout
            if pending:
                # manually cancel tasks since asyncio.wait will not
                for remaining_task in pending:
                    remaining_task.cancel()
                logger.debug(f"Execution of {len(pending)} collectors timed out.")

    async def shutdown(self) -> None:
        """Release resources for all registered collectors"""
        logger.debug(f"{type(self).__name__} shutting down collectors...")
        if list(self.all_collectors):
            shutdown_tasks = []
            # create an async tasks to execute all shutdowns in parallel
            for item in self.all_collectors:
                shutdown_tasks.append(asyncio.create_task(item.shutdown()))
            # await until all shutdowns are complete
            await asyncio.wait(shutdown_tasks)
        logger.debug("Collector shutdown complete...")

    @property
    def all_collectors(self) -> t.Sequence[Collector]:
        """Get a list of all registered collectors

        :return: a collection of registered collectors for all entities
        """
        # flatten and return all the lists-of-collectors that are registered
        collectors = itertools.chain.from_iterable(self._collectors.values())
        return [collector for collector in collectors if collector.enabled]

    @property
    def dead_collectors(self) -> t.Sequence[Collector]:
        """Get a list of all disabled collectors

        :return: a collection of disabled collectors for all entities
        """
        collectors = itertools.chain.from_iterable(self._collectors.values())
        return [collector for collector in collectors if not collector.enabled]

    def register_collectors(self, entity: JobEntity) -> None:
        """Find all configured collectors for the entity and register them

        :param entity: a `JobEntity` instance that will have all configured collectors
        registered for collection. Configuration is found in the `RuntimeManifest`
        """
        collectors: t.List[Collector] = []

        # ONLY db telemetry is implemented at this time. This resolver must
        # be updated when non-database or always-on collectors are introduced
        if entity.is_db and entity.telemetry_on:
            if mem_out := entity.collectors.get("memory", None):
                collectors.append(DBMemoryCollector(entity, FileSink(mem_out)))

            if con_out := entity.collectors.get("client", None):
                collectors.append(DBConnectionCollector(entity, FileSink(con_out)))

            if num_out := entity.collectors.get("client_count", None):
                collectors.append(DBConnectionCountCollector(entity, FileSink(num_out)))
        else:
            logger.debug(f"Collectors disabled for db {entity.name}")

        self.add_all(collectors)

    def register_all_collectors(self, entities: t.Sequence[JobEntity]) -> None:
        """Find all configured collectors for the entity and register them

        :param entities: entities to call `register_collectors` for
        """
        for entity in entities:
            self.register_collectors(entity)
