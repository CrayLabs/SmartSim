# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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
# import pathlib

import typing as t

import pytest

import smartsim._core.entrypoints.telemetry_monitor
import smartsim._core.utils.telemetry.collector
from conftest import MockCollectorEntityFunc, MockSink
from smartsim._core.utils.telemetry.collector import (
    DBConnectionCollector,
    DBConnectionCountCollector,
    DBMemoryCollector,
    redisa,
)

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

PrepareFS = t.Callable[[dict], smartsim.experiment.FeatureStore]


@pytest.mark.asyncio
async def test_dbmemcollector_prepare(
    mock_entity: MockCollectorEntityFunc, mock_sink
) -> None:
    """Ensure that collector preparation succeeds when expected"""
    entity = mock_entity(telemetry_on=True)

    collector = DBMemoryCollector(entity, mock_sink())
    await collector.prepare()
    assert collector._client


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail(
    mock_entity: MockCollectorEntityFunc,
    mock_sink: MockSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that collector preparation reports a failure to connect
    when the redis client cannot be created"""
    entity = mock_entity(telemetry_on=True)

    with monkeypatch.context() as ctx:
        # mock up a redis constructor that returns None
        ctx.setattr(redisa, "Redis", lambda host, port: None)

        sink = mock_sink()
        collector = DBMemoryCollector(entity, sink)
        assert sink.num_saves == 0

        await collector.prepare()

        # Attempt to save header when preparing...
        assert not collector._client
        assert sink.num_saves == 1


@pytest.mark.asyncio
async def test_dbcollector_config(
    mock_entity: MockCollectorEntityFunc,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that missing required db collector config causes an exception"""

    # Check that a bad host causes exception
    entity = mock_entity(host="", telemetry_on=True)
    with pytest.raises(ValueError):
        DBMemoryCollector(entity, mock_sink())

    entity = mock_entity(host="   ", telemetry_on=True)
    with pytest.raises(ValueError):
        DBMemoryCollector(entity, mock_sink())

    # Check that a bad port causes exception
    entity = mock_entity(port="", telemetry_on=True)  # type: ignore
    with pytest.raises(ValueError):
        DBMemoryCollector(entity, mock_sink())


@pytest.mark.asyncio
async def test_dbmemcollector_prepare_fail_dep(
    mock_entity: MockCollectorEntityFunc,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[t.Any],
) -> None:
    """Ensure that collector preparation attempts to connect, ensure it
    reports a failure if the db conn bombs"""
    entity = mock_entity(telemetry_on=True)

    def raiser(*args: t.Any, **kwargs: t.Any) -> None:
        # mock raising exception on connect attempts to test err handling
        raise redisa.ConnectionError("mock connection failure")

    sink = mock_sink()
    collector = DBMemoryCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redisa, "Redis", raiser)

        assert sink.num_saves == 0
        await collector.prepare()

        assert sink.num_saves == 1
        assert not collector._client


@pytest.mark.asyncio
async def test_dbmemcollector_collect(
    mock_entity: MockCollectorEntityFunc,
    mock_redis,
    mock_mem,
    mock_sink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity(telemetry_on=True)

    sink = mock_sink()
    collector = DBMemoryCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redisa, "Redis", mock_redis(mem_stats=mock_mem(1, 2)))
        ctx.setattr(
            smartsim._core.utils.telemetry.collector,
            "get_ts_ms",
            lambda: 12131415,
        )

        await collector.prepare()
        await collector.collect()

        reqd_items = {
            "timestamp",
            "total_system_memory",
            "used_memory",
            "used_memory_peak",
        }
        actual_items = set(sink.args)

        reqd_values = {12131415, 1000.0, 1111.0, 1234.0}
        actual_values = set(sink.args)
        assert actual_values == reqd_values


@pytest.mark.asyncio
async def test_dbmemcollector_integration(
    mock_entity: MockCollectorEntityFunc,
    mock_sink: MockSink,
    prepare_fs: PrepareFS,
    local_fs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration test with a real feature store instance to ensure
    output data matches expectations and proper db client API uage"""

    fs = prepare_fs(local_fs).featurestore
    entity = mock_entity(port=fs.ports[0], telemetry_on=True)

    sink = mock_sink()
    collector = DBMemoryCollector(entity, sink)

    with monkeypatch.context() as ctx:
        ctx.setattr(
            smartsim._core.utils.telemetry.collector,
            "get_ts_ms",
            lambda: 12131415,
        )
        assert sink.num_saves == 0
        await collector.prepare()
        assert sink.num_saves == 1
        await collector.collect()
        assert sink.num_saves == 2

        stats = sink.args
        assert len(stats) == 4  # show we have the expected amount of data points
        ts = 12131415

        assert ts in stats


@pytest.mark.asyncio
async def test_dbconncollector_collect(
    mock_entity: MockCollectorEntityFunc,
    mock_sink,
    mock_redis,
    mock_con,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity(telemetry_on=True)

    sink = mock_sink()
    collector = DBConnectionCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redisa, "Redis", mock_redis(client_stats=mock_con(1, 2)))

        assert sink.num_saves == 0
        await collector.prepare()
        assert sink.num_saves == 1
        await collector.collect()
        assert sink.num_saves == 3  # save twice w/two datapoints

        stats = sink.args

        idx = 1
        id0, ip0 = f"ABC{idx}", f"127.0.0.{idx}:1234"
        id1, ip1 = f"XYZ{idx}", f"127.0.0.{idx}:2345"
        exp_clients = [{"id": id0, "addr": ip0}, {"id": id1, "addr": ip1}]

        assert len(exp_clients) + 1 == len(stats)  # output includes timestamp
        assert id0 in set(client["id"] for client in exp_clients)
        assert id1 in set(client["id"] for client in exp_clients)
        assert ip0 in set(client["addr"] for client in exp_clients)
        assert ip1 in set(client["addr"] for client in exp_clients)


@pytest.mark.asyncio
async def test_dbconn_count_collector_collect(
    mock_entity: MockCollectorEntityFunc,
    mock_sink,
    mock_redis,
    mock_con,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that a valid response is returned as expected"""
    entity = mock_entity(telemetry_on=True)

    sink = mock_sink()
    collector = DBConnectionCountCollector(entity, sink)
    with monkeypatch.context() as ctx:
        ctx.setattr(redisa, "Redis", mock_redis(client_stats=mock_con(1, 2)))

        assert sink.num_saves == 0
        await collector.prepare()
        assert sink.num_saves == 1
        await collector.collect()
        assert sink.num_saves == 2

        stats = sink.args
        exp_counts = 2

        assert exp_counts == len(stats)  # output includes timestamp


@pytest.mark.asyncio
async def test_dbconncollector_integration(
    mock_entity: MockCollectorEntityFunc,
    mock_sink: MockSink,
    prepare_fs: PrepareFS,
    local_fs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration test with a real feature store instance to ensure
    output data matches expectations and proper db client API uage"""

    fs = prepare_fs(local_fs).featurestore
    entity = mock_entity(port=fs.ports[0], telemetry_on=True)

    sink = mock_sink()
    collector = DBConnectionCollector(entity, sink)

    with monkeypatch.context() as ctx:
        ctx.setattr(
            smartsim._core.utils.telemetry.collector,
            "get_ts_ms",
            lambda: 12131415,
        )
        await collector.prepare()
        await collector.collect()
        stats = sink.args

        ip = "127.0.0.1:"
        num_conns = int(stats[1])
        ts = 12131415

        assert ts in stats
        assert num_conns > 0
        assert ip in stats[2]
