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
import typing as t
import uuid

import pytest

from smartsim._core.entrypoints.telemetrymonitor import (
    DbConnectionCollector,
    DbMemoryCollector,
    JobEntity,
)
from smartsim._core.entrypoints.telemetrymonitor import redis as tredis

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.group_a


@pytest.fixture
def mock_entity(test_dir):
    def _mock_entity(
        host: str = "127.0.0.1", port: str = "6379", name: str = "", type: str = ""
    ):
        entity = JobEntity()
        entity.name = name if name else str(uuid.uuid4())
        entity.status_dir = test_dir
        entity.type = type
        entity.telemetry_on = True
        entity.collectors = {
            "client": "",
            "client_count": "",
            "memory": "",
        }
        entity.config = {
            "host": host,
            "port": port,
        }
        return entity

    return _mock_entity


@pytest.mark.asyncio
async def test_dbmemcollector_prepare(mock_entity, mock_sink):
    """Ensure that collector preparation succeeds when expected"""
    entity = mock_entity()

    collector = DbMemoryCollector(entity, mock_sink())
    await collector.prepare()
    assert collector._client


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail(
    mock_entity,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """Ensure that collector preparation reports a failure to connect
    when the redis client cannot be created"""
    entity = mock_entity()

    with monkeypatch.context() as ctx:
        # mock up a redis constructor that returns None
        ctx.setattr(tredis, "Redis", lambda host, port: None)

        capsys.readouterr()  # clear capture
        collector = DbMemoryCollector(entity, mock_sink())
        await collector.prepare()

        assert not collector._client
        assert collector.value is None


@pytest.mark.asyncio
async def test_dbcollector_config(
    mock_entity,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ensure that missing required db collector config causes an exception"""

    # Check that a bad host causes exception
    entity = mock_entity(host="")
    with pytest.raises(ValueError):
        DbMemoryCollector(entity, mock_sink())

    entity = mock_entity(host="   ")
    with pytest.raises(ValueError):
        DbMemoryCollector(entity, mock_sink())

    # Check that a bad port causes exception
    entity = mock_entity(port="")
    with pytest.raises(ValueError):
        DbMemoryCollector(entity, mock_sink())


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail_dep(
    mock_entity,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """Ensure that collector preparation attempts to connect, ensure it
    reports a failure if the db conn bombs"""
    entity = mock_entity()

    def raiser(*args, **kwargs):
        # mock raising exception on connect attempts to test err handling
        raise tredis.ConnectionError("mock connection failure")

    capsys.readouterr()  # clear capture
    collector = DbMemoryCollector(entity, mock_sink())
    with monkeypatch.context() as ctx:
        ctx.setattr(tredis, "Redis", raiser)
        # with pytest.raises(SmartSimError) as ex:
        await collector.prepare()

        capture = capsys.readouterr()  # retrieve logs for operation
        assert not collector._client
        assert collector.value is None


@pytest.mark.asyncio
async def test_dbmemcollector_collect(
    mock_entity, mock_redis, mock_mem, mock_sink, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    sink = mock_sink()
    collector = DbMemoryCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(tredis, "Redis", mock_redis(mem_stats=mock_mem(1, 2)))

        await collector.prepare()
        await collector.collect()
        # stats = collector.value

        reqd_items = set(
            ("timestamp", "total_system_memory", "used_memory", "used_memory_peak")
        )
        actual_items = set(sink.args)

        reqd_values = set((1000, 1111, 1234))
        actual_values = set(sink.args.values())
        assert actual_items.issuperset(reqd_items)
        assert actual_values.issuperset(reqd_values)


@pytest.mark.asyncio
async def test_dbmemcollector_integration(mock_entity, mock_sink, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbMemoryCollector(entity, mock_sink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert (
        len(stats) == 3
    )  # prove we filtered to expected data size, no timestamp in data.
    assert stats["used_memory"] > 0  # prove used_memory was retrieved
    assert stats["used_memory_peak"] > 0  # prove used_memory_peak was retrieved
    assert stats["total_system_memory"] > 0  # prove total_system_memory was retrieved


@pytest.mark.asyncio
async def test_dbconncollector_collect(
    mock_entity, mock_sink, mock_redis, mock_con, monkeypatch: pytest.MonkeyPatch
):
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity()

    collector = DbConnectionCollector(entity, mock_sink())
    with monkeypatch.context() as ctx:
        ctx.setattr(tredis, "Redis", mock_redis(client_stats=mock_con(1, 2)))

        await collector.prepare()
        await collector.collect()

        stats = collector.value

        idx = 1
        id0, ip0 = f"ABC{idx}", f"127.0.0.{idx}:1234"
        id1, ip1 = f"XYZ{idx}", f"127.0.0.{idx}:2345"
        exp_clients = [{"id": id0, "addr": ip0}, {"id": id1, "addr": ip1}]

        assert len(exp_clients) == len(stats)
        assert id0 in set(client["id"] for client in exp_clients)
        assert id1 in set(client["id"] for client in exp_clients)
        assert ip0 in set(client["addr"] for client in exp_clients)
        assert ip1 in set(client["addr"] for client in exp_clients)


@pytest.mark.asyncio
async def test_dbconncollector_integration(mock_entity, mock_sink, local_db):
    """Integration test with a real orchestrator instance to ensure
    output data matches expectations and proper db client API uage"""
    entity = mock_entity(port=local_db.ports[0])

    collector = DbConnectionCollector(entity, mock_sink())

    await collector.prepare()
    await collector.collect()
    stats = collector.value

    assert len(stats) > 0
    assert stats[0]
