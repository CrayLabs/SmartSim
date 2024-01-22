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
import copy
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
)
from smartsim.error import SmartSimError

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


class MockSink(Sink):
    """Telemetry sink that writes console output for testing purposes"""
    def save(self, **kwargs: t.Any) -> None:
        """Save all arguments as console logged messages"""
        print(f"MockSink received args: {kwargs}")
        self.args = kwargs


@pytest.fixture
def mock_redis():
    def _mock_redis(
        is_conn: bool = True, conn_side_effect=None, mem_stats=None, client_stats=None
    ):
        class MockConn:
            def __init__(self, *args, **kwargs) -> None:
                if conn_side_effect is not None:
                    conn_side_effect()

            async def info(self) -> t.Dict[str, t.Any]:
                return mem_stats

            async def client_list(self) -> t.Dict[str, t.Any]:
                return client_stats

        return MockConn

    return _mock_redis


@pytest.fixture
def mock_entity():
    def _mock_entity(host: str = "127.0.0.1", port: str = "6379", name: str = ""):
        entity = JobEntity()
        entity.name = name if name else str(uuid.uuid4())
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
    mock_entity, mock_redis, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    collector = DbMemoryCollector(entity, MockSink())
    with monkeypatch.context() as ctx:
        m1, m2, m3 = 12345, 23456, 34567
        mock_data = {
            "used_memory": m1,
            "used_memory_peak": m2,
            "total_system_memory": m3,
        }
        ctx.setattr(redis, "Redis", mock_redis(mem_stats=mock_data))

        await collector.prepare()
        await collector.collect()
        stats = collector.value

        assert set(mock_data.values()) == set(stats.values())


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
    mock_entity, mock_redis, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    collector = DbConnectionCollector(entity, MockSink())
    with monkeypatch.context() as ctx:
        a1, a2 = "127.0.0.1:1234", "127.0.0.1:2345"
        mock_data = [
            {
                "addr": a1,
            },
            {
                "addr": a2,
            },
        ]
        ctx.setattr(redis, "Redis", mock_redis(client_stats=mock_data))

        await collector.prepare()
        await collector.collect()

        stats = collector.value

        assert set(x["addr"] for x in mock_data) == set(stats)


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
async def test_collector_manager_collect(mock_entity, local_db):
    """Ensure that all collectors are executed and some metric is retrieved"""
    entity1 = mock_entity(port=local_db.ports[0])
    entity2 = mock_entity(port=local_db.ports[0])

    # todo: consider a MockSink so i don't have to save the last value in the collector
    s1, s2, s3 = MockSink(), MockSink(), MockSink()
    con_col1 = DbConnectionCollector(entity1, s1)
    mem_col1 = DbMemoryCollector(entity1, s2)
    mem_col2 = DbMemoryCollector(entity2, s3)

    manager = CollectorManager()
    manager.add_all([con_col1, mem_col1, mem_col2])

    # Execute collection
    await manager.collect()

    # verify each collector retrieved some metric & sent it to the sink
    for collector in manager.all_collectors:
        value = t.cast(MockSink, collector.sink).args
        assert value is not None and value
