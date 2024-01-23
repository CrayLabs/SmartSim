# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import asyncio
import datetime
import typing as t
import uuid

import pytest

from smartsim._core.entrypoints.telemetrymonitor import (
    CollectorManager,
    DbConnectionCollector,
    DbMemoryCollector,
    JobEntity,
    redis,
    Sink,
    FileSink,
)
from smartsim.error import SmartSimError

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


class MockSink(Sink):
    """Telemetry sink that writes console output for testing purposes"""

    def __init__(self, delay_ms: int = 0):
        self._delay_ms = delay_ms

    async def save(self, **kwargs: t.Any) -> None:
        """Save all arguments as console logged messages"""
        print(f"MockSink received args: {kwargs}")
        if self._delay_ms:
            # mimic slow collection....
            delay_s = self._delay_ms / 1000
            await asyncio.sleep(delay_s)
        self.args = kwargs


@pytest.fixture
def mock_con():
    def _mock_con(min=1, max=1000):
        i = min
        while True:
            yield [{"addr": f"127.0.0.{i}:1234"}, {"addr": f"127.0.0.{i}:2345"}]
            i += 1
            if i > max:
                return None

    return _mock_con


@pytest.fixture
def mock_mem():
    def _mock_mem(min=1, max=1000):
        i = min
        while True:
            yield {
                "total_system_memory": 1000 * i,
                "used_memory": 1111 * i,
                "used_memory_peak": 1234 * i,
            }
            i += 1
            if i > max:
                return None

    return _mock_mem


@pytest.fixture
def mock_redis():
    def _mock_redis(
        is_conn: bool = True,
        conn_side_effect=None,
        mem_stats=None,
        client_stats=None,
        coll_side_effect=None,
    ):
        class MockConn:
            def __init__(self, *args, **kwargs) -> None:
                if conn_side_effect is not None:
                    conn_side_effect()

            async def info(self) -> t.Dict[str, t.Any]:
                if coll_side_effect:
                    await coll_side_effect()

                if mem_stats:
                    return next(mem_stats)
                return {
                    "ts": 111,
                    "total_system_memory": "111",
                    "used_memory": "222",
                    "used_memory_peak": "333",
                }

            async def client_list(self) -> t.Dict[str, t.Any]:
                if coll_side_effect:
                    await coll_side_effect()

                if client_stats:
                    return next(client_stats)
                return {"ts": 111, "addr": "127.0.0.1"}

        return MockConn

    return _mock_redis


@pytest.fixture
def mock_entity(test_dir):
    def _mock_entity(host: str = "127.0.0.1", port: str = "6379", name: str = ""):
        entity = JobEntity()
        entity.name = name if name else str(uuid.uuid4())
        entity.status_dir = test_dir
        entity.meta = {
            "host": host,
            "port": port,
        }
        return entity

    return _mock_entity


@pytest.mark.asyncio
async def test_dbmemcollector_prepare(mock_entity):
    """Ensure that collector preparation succeeds when expected"""
    entity = mock_entity()

    collector = DbMemoryCollector(entity, MockSink())
    await collector.prepare()
    assert collector._client


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail(
    mock_entity, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that collector preparation reports a failure to connect"""
    entity = mock_entity()

    with monkeypatch.context() as ctx:
        # mock up a redis constructor that returns None
        ctx.setattr(redis, "Redis", lambda host, port: None)

        with pytest.raises(SmartSimError) as ex:
            collector = DbMemoryCollector(entity, MockSink())
            await collector.prepare()

        assert not collector._client

        err_content = ",".join(ex.value.args)
        assert "connect" in err_content


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail_dep(
    mock_entity, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that collector preparation attempts to connect, ensure it
    reports a failure if the db conn bombs"""
    entity = mock_entity()

    def raiser():
        # mock raising exception on connect attempts to test err handling
        raise redis.ConnectionError("mock connection failure")

    collector = DbMemoryCollector(entity, MockSink())
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", raiser)
        with pytest.raises(SmartSimError) as ex:
            await collector.prepare()

        assert not collector._client

        err_content = ",".join(ex.value.args)
        assert "communicate" in err_content


@pytest.mark.asyncio
async def test_dbmemcollector_collect(
    mock_entity, mock_redis, mock_mem, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    sink = MockSink()
    collector = DbMemoryCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", mock_redis(mem_stats=mock_mem(1, 2)))

        await collector.prepare()
        await collector.collect()
        stats = collector.value

        assert set(
            ("ts", "total_system_memory", "used_memory", "used_memory_peak")
        ) == set(sink.args)
        assert set((1000, 1111, 1234)).issubset(set(sink.args.values()))


@pytest.mark.asyncio
async def test_dbmemcollector_integration(mock_entity, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbMemoryCollector(entity, MockSink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert len(stats) == 3  # prove we filtered to expected data size
    assert stats["used_memory"] > 0  # prove used_memory was retrieved
    assert stats["used_memory_peak"] > 0  # prove used_memory_peak was retrieved
    assert stats["total_system_memory"] > 0  # prove total_system_memory was retrieved


@pytest.mark.asyncio
async def test_dbconncollector_collect(
    mock_entity, mock_redis, mock_con, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    collector = DbConnectionCollector(entity, MockSink())
    with monkeypatch.context() as ctx:
        ctx.setattr(redis, "Redis", mock_redis(client_stats=mock_con(1, 2)))

        await collector.prepare()
        await collector.collect()

        stats = collector.value

        assert set(["127.0.0.1:1234", "127.0.0.1:2345"]) == set(stats)


@pytest.mark.asyncio
async def test_dbconncollector_integration(mock_entity, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbConnectionCollector(entity, MockSink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert len(stats) == 1
    assert stats[0]


def test_collector_manager_add(mock_entity):
    """Ensure that collector manager add & clear work as expected"""
    entity1 = mock_entity()

    con_col = DbConnectionCollector(entity1, MockSink())
    mem_col = DbMemoryCollector(entity1, MockSink())

    manager = CollectorManager()

    # ensure manager starts empty
    assert len(manager.all_collectors) == 0

    # ensure added item is in the collector list
    manager.add(con_col)
    assert len(manager.all_collectors) == 1

    # ensure a duplicate isn't added
    manager.add(con_col)
    assert len(manager.all_collectors) == 1

    # ensure another collector for the same entity is added
    manager.add(mem_col)
    assert len(manager.all_collectors) == 2

    # create a collector for another entity
    entity2 = mock_entity()
    con_col2 = DbConnectionCollector(entity2, MockSink())

    # ensure collectors w/same type for new entities are not treated as dupes
    manager.add(con_col2)
    assert len(manager.all_collectors) == 3

    # verify no dupe on second entity
    manager.add(con_col2)
    assert len(manager.all_collectors) == 3

    manager.clear()
    assert len(manager.all_collectors) == 0

    # ensure post-clear adding still works
    manager.add(con_col2)
    assert len(manager.all_collectors) == 1


def test_collector_manager_add_multi(mock_entity):
    """Ensure that collector manager multi-add works as expected"""
    entity = mock_entity()

    con_col = DbConnectionCollector(entity, MockSink())
    mem_col = DbMemoryCollector(entity, MockSink())
    manager = CollectorManager()

    # add multiple items at once
    manager.add_all([con_col, mem_col])

    assert len(manager.all_collectors) == 2

    # ensure multi-add does not produce dupes
    con_col2 = DbConnectionCollector(entity, MockSink())
    mem_col2 = DbMemoryCollector(entity, MockSink())

    manager.add_all([con_col2, mem_col2])
    assert len(manager.all_collectors) == 2


@pytest.mark.asyncio
async def test_collector_manager_collect(
    mock_entity, mock_redis, monkeypatch, mock_con, mock_mem
):
    """Ensure that all collectors are executed and some metric is retrieved
    NOTE: responses & producer are mocked"""
    entity1 = mock_entity(port=1234, name="entity1")
    entity2 = mock_entity(port=2345, name="entity2")

    sinks = [MockSink(), MockSink(), MockSink()]
    con_col1 = DbConnectionCollector(entity1, sinks[0])
    mem_col1 = DbMemoryCollector(entity1, sinks[1])
    mem_col2 = DbMemoryCollector(entity2, sinks[2])

    manager = CollectorManager()
    manager.add_all([con_col1, mem_col1, mem_col2])

    # Execute collection
    with monkeypatch.context() as ctx:
        ctx.setattr(
            redis,
            "Redis",
            mock_redis(client_stats=mock_con(1, 10), mem_stats=mock_mem(1, 10)),
        )
        await manager.collect()

    # verify each collector retrieved some metric & sent it to the sink
    for sink in sinks:
        value = sink.args
        assert value is not None and value


@pytest.mark.asyncio
async def test_collector_manager_collect_filesink(
    mock_entity, mock_redis, monkeypatch, mock_mem, mock_con
):
    """Ensure that all collectors are executed and some metric is retrieved
    and the FileSink is written to as expected"""
    entity1 = mock_entity(port=1234, name="entity1")
    entity2 = mock_entity(port=2345, name="entity2")

    sinks = [
        FileSink(entity1, "1_con.csv"),
        FileSink(entity1, "1_mem.csv"),
        FileSink(entity2, "2_mem.csv"),
    ]
    con_col1 = DbConnectionCollector(entity1, sinks[0])
    mem_col1 = DbMemoryCollector(entity1, sinks[1])
    mem_col2 = DbMemoryCollector(entity2, sinks[2])

    manager = CollectorManager()
    manager.add_all([con_col1, mem_col1, mem_col2])

    # Execute collection
    with monkeypatch.context() as ctx:
        ctx.setattr(
            redis,
            "Redis",
            mock_redis(client_stats=mock_con(1, 10), mem_stats=mock_mem(1, 10)),
        )
        await manager.collect()

    # verify each collector retrieved some metric & sent it to the sink
    for sink in sinks:
        save_to = sink.path
        assert save_to.exists()
        if "con" in str(save_to):
            assert "127.0.0." in save_to.read_text()
        else:
            # look for something multiplied by 1000
            assert "000" in save_to.read_text()


@pytest.mark.asyncio
async def test_collector_manager_collect_integration(mock_entity, local_db):
    """Ensure that all collectors are executed and some metric is retrieved"""
    entity1 = mock_entity(port=local_db.ports[0], name="e1")
    entity2 = mock_entity(port=local_db.ports[0], name="e2")

    # todo: consider a MockSink so i don't have to save the last value in the collector
    sinks = [MockSink(), MockSink(), MockSink()]
    con_col1 = DbConnectionCollector(entity1, sinks[0])
    mem_col1 = DbMemoryCollector(entity1, sinks[1])
    mem_col2 = DbMemoryCollector(entity2, sinks[2])

    manager = CollectorManager()
    manager.add_all([con_col1, mem_col1, mem_col2])

    # Execute collection
    await manager.collect()

    # verify each collector retrieved some metric & sent it to the sink
    for sink in sinks:
        value = sink.args
        assert value is not None and value


@pytest.mark.parametrize(
    "timeout_at,delay_for,expect_fail",
    [
        pytest.param(1000, 5000, True, id="1s timeout"),
        pytest.param(2000, 5000, True, id="2s timeout"),
        pytest.param(3000, 5000, True, id="3s timeout"),
        pytest.param(4000, 5000, True, id="4s timeout"),
        pytest.param(2000, 1000, False, id="under timeout"),
    ],
)
@pytest.mark.asyncio
async def test_collector_timeout(
    mock_entity,
    mock_redis,
    monkeypatch: pytest.MonkeyPatch,
    mock_mem,
    mock_con,
    timeout_at,
    delay_for,
    expect_fail,
):
    """Ensure that the collector timeout is honored"""
    entity1 = mock_entity(port=1234, name="e1")
    entity2 = mock_entity(port=2345, name="e2")

    sinks = [MockSink(), MockSink(), MockSink()]
    con_col1 = DbConnectionCollector(entity1, sinks[0])
    mem_col1 = DbMemoryCollector(entity1, sinks[1])
    mem_col2 = DbMemoryCollector(entity2, sinks[2])

    manager = CollectorManager(timeout_ms=timeout_at)
    manager.add_all([con_col1, mem_col1, mem_col2])

    async def snooze():
        await asyncio.sleep(delay_for / 1000)

    # Execute collection
    with monkeypatch.context() as ctx:
        ctx.setattr(
            redis,
            "Redis",
            mock_redis(
                client_stats=mock_con(1, 10),
                mem_stats=mock_mem(1, 10),
                coll_side_effect=snooze,
            ),
        )

        ts0 = datetime.datetime.utcnow()
        await manager.collect()
        ts1 = datetime.datetime.utcnow()

        t_diff = ts1 - ts0
        actual_delay = 1000 * t_diff.seconds

        if expect_fail:    
            assert timeout_at <= actual_delay < delay_for
        else:
            assert delay_for <= actual_delay < timeout_at
