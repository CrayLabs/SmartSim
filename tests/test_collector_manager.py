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

import pytest

from conftest import MockCollectorEntityFunc
from smartsim._core.entrypoints.telemetrymonitor import (
    CollectorManager,
    DbConnectionCollector,
    DbConnectionCountCollector,
    DbMemoryCollector,
    FileSink,
    JobEntity,
    redis,
)

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


def test_collector_manager_add(
    mock_entity: MockCollectorEntityFunc, mock_sink
) -> None:
    """Ensure that collector manager add & clear work as expected"""
    entity1 = mock_entity()

    con_col = DbConnectionCollector(entity1, mock_sink())
    mem_col = DbMemoryCollector(entity1, mock_sink())

    manager = CollectorManager()

    # ensure manager starts empty
    assert len(list(manager.all_collectors)) == 0

    # ensure added item is in the collector list
    manager.add(con_col)
    assert len(list(manager.all_collectors)) == 1

    # ensure a duplicate isn't added
    manager.add(con_col)
    assert len(list(manager.all_collectors)) == 1

    # ensure another collector for the same entity is added
    manager.add(mem_col)
    assert len(list(manager.all_collectors)) == 2

    # create a collector for another entity
    entity2 = mock_entity()
    con_col2 = DbConnectionCollector(entity2, mock_sink())

    # ensure collectors w/same type for new entities are not treated as dupes
    manager.add(con_col2)
    assert len(list(manager.all_collectors)) == 3

    # verify no dupe on second entity
    manager.add(con_col2)
    assert len(list(manager.all_collectors)) == 3

    manager.clear()
    assert len(list(manager.all_collectors)) == 0

    # ensure post-clear adding still works
    manager.add(con_col2)
    assert len(list(manager.all_collectors)) == 1


def test_collector_manager_add_multi(
    mock_entity: MockCollectorEntityFunc, mock_sink
) -> None:
    """Ensure that collector manager multi-add works as expected"""
    entity = mock_entity()

    con_col = DbConnectionCollector(entity, mock_sink())
    mem_col = DbMemoryCollector(entity, mock_sink())
    manager = CollectorManager()

    # add multiple items at once
    manager.add_all([con_col, mem_col])

    assert len(list(manager.all_collectors)) == 2

    # ensure multi-add does not produce dupes
    con_col2 = DbConnectionCollector(entity, mock_sink())
    mem_col2 = DbMemoryCollector(entity, mock_sink())

    manager.add_all([con_col2, mem_col2])
    assert len(list(manager.all_collectors)) == 2


@pytest.mark.asyncio
async def test_collector_manager_collect(
    mock_entity: MockCollectorEntityFunc,
    mock_redis,
    monkeypatch: pytest.MonkeyPatch,
    mock_con,
    mock_mem,
    mock_sink,
) -> None:
    """Ensure that all collectors are executed and some metric is retrieved
    NOTE: responses & producer are mocked"""
    entity1 = mock_entity(port=1234, name="entity1")
    entity2 = mock_entity(port=2345, name="entity2")

    sinks = [mock_sink(), mock_sink(), mock_sink()]
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
    mock_entity: MockCollectorEntityFunc,
    mock_redis,
    monkeypatch,
    mock_mem,
    mock_con,
) -> None:
    """Ensure that all collectors are executed and some metric is retrieved
    and the FileSink is written to as expected"""
    entity1 = mock_entity(port=1234, name="entity1")
    entity2 = mock_entity(port=2345, name="entity2")

    sinks = [
        FileSink(entity1.status_dir + "/1_con.csv"),
        FileSink(entity1.status_dir + "/1_mem.csv"),
        FileSink(entity2.status_dir + "/2_mem.csv"),
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
async def test_collector_manager_collect_integration(
    test_dir: str, mock_entity: MockCollectorEntityFunc, local_db, mock_sink
) -> None:
    """Ensure that all collectors are executed and some metric is retrieved"""
    entity1 = mock_entity(port=local_db.ports[0], name="e1")
    entity2 = mock_entity(port=local_db.ports[0], name="e2")

    # todo: consider a MockSink so i don't have to save the last value in the collector
    sinks = [mock_sink(), mock_sink(), mock_sink()]
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
async def test_collector_manager_timeout(
    mock_entity: MockCollectorEntityFunc,
    mock_redis,
    monkeypatch: pytest.MonkeyPatch,
    mock_mem,
    mock_con,
    timeout_at: int,
    delay_for: int,
    expect_fail: bool,
    mock_sink,
) -> None:
    """Ensure that the collector timeout is honored"""
    entity1 = mock_entity(port=1234, name="e1")
    entity2 = mock_entity(port=2345, name="e2")

    sinks = [mock_sink(), mock_sink(), mock_sink()]
    con_col1 = DbConnectionCollector(entity1, sinks[0])
    mem_col1 = DbMemoryCollector(entity1, sinks[1])
    mem_col2 = DbMemoryCollector(entity2, sinks[2])

    manager = CollectorManager(timeout_ms=timeout_at)
    manager.add_all([con_col1, mem_col1, mem_col2])

    async def snooze() -> None:
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


@pytest.mark.asyncio
async def test_collector_manager_find(
    mock_entity: MockCollectorEntityFunc
) -> None:
    """Ensure that the manifest allows individually enabling a given collector"""
    entity: JobEntity = mock_entity(port=1234, name="entity1", type="orchestrator")
    manager = CollectorManager()

    # 0. popping all should result in no collectors mapping to the entity
    found = manager.find_collectors(entity)
    assert len(found) == 0

    # 1. ensure DbConnectionCollector is mapped
    entity = mock_entity(port=1234, name="entity1", type="orchestrator")
    entity.collectors["client"] = "mock/path.csv"
    manager = CollectorManager()

    # 2. client count collector should be mapped
    found = manager.find_collectors(entity)
    assert len(found) == 1
    assert isinstance(found[0], DbConnectionCollector)

    # 3. ensure DbConnectionCountCollector is mapped
    entity = mock_entity(port=1234, name="entity1", type="orchestrator")
    entity.collectors["client_count"] = "mock/path.csv"
    manager = CollectorManager()

    # 3. client count collector should be mapped
    found = manager.find_collectors(entity)
    assert len(found) == 1
    assert isinstance(found[0], DbConnectionCountCollector)

    # ensure DbMemoryCollector is mapped
    entity = mock_entity(port=1234, name="entity1", type="orchestrator")
    entity.collectors["memory"] = "mock/path.csv"
    manager = CollectorManager()

    # client count collector should be mapped
    found = manager.find_collectors(entity)
    assert len(found) == 1
    assert isinstance(found[0], DbMemoryCollector)


@pytest.mark.asyncio
async def test_collector_manager_find_entity_disabled(
    mock_entity: MockCollectorEntityFunc
) -> None:
    """Ensure that disabling telemetry on the entity results in no collectors"""
    entity: JobEntity = mock_entity(port=1234, name="entity1", type="orchestrator")
    manager = CollectorManager()

    # set paths for all known collectors
    entity.collectors["client"] = "mock/path.csv"
    entity.collectors["client_count"] = "mock/path.csv"
    entity.collectors["memory"] = "mock/path.csv"

    manager = CollectorManager()

    # ON behavior should locate multiple collectors
    entity.telemetry_on = True
    found = manager.find_collectors(entity)
    assert len(found) > 0

    # OFF behavior should locate ZERO collectors
    entity.telemetry_on = False
    found = manager.find_collectors(entity)
    assert len(found) == 0


@pytest.mark.asyncio
async def test_collector_manager_find_entity_unmapped(
    mock_entity: MockCollectorEntityFunc
) -> None:
    """Ensure that an entity type that is not mapped results in no collectors"""
    entity: JobEntity = mock_entity(port=1234, name="entity1", type="model")
    manager = CollectorManager()

    # set paths for all known collectors
    entity.collectors["client"] = "mock/path.csv"
    entity.collectors["client_count"] = "mock/path.csv"
    entity.collectors["memory"] = "mock/path.csv"

    manager = CollectorManager()

    # ON behavior should locate ZERO collectors
    entity.telemetry_on = True
    found = manager.find_collectors(entity)
    assert len(found) == 0

    # OFF behavior should locate ZERO collectors
    entity.telemetry_on = False
    found = manager.find_collectors(entity)
    assert len(found) == 0
