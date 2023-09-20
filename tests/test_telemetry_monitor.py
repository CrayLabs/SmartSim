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


import logging
import pathlib
import psutil
import pytest
import re
import sys
import typing as t
import uuid
from conftest import FileUtils
from smartsim._core.control.job import Job

from smartsim.telemetrymonitor import (
    get_parser,
    get_ts,
    main,
    track_event,
    PersistableEntity,
    track_started,
    track_completed,
    track_timestep,
    load_manifest,
)


ALL_ARGS = {"-d", "-f"}
logger = logging.getLogger()


@pytest.mark.parametrize(
    ["cmd", "missing"],
    [
        pytest.param("", {"-d", "-f"}, id="no args"),
        pytest.param("-d /foo/bar", {"-f"}, id="no freq"),
        pytest.param("-f 123", {"-d"}, id="no dir"),
    ],
)
def test_parser_reqd_args(capsys, cmd, missing):
    """Test that the parser reports any missing required arguments"""
    parser = get_parser()

    args = cmd.split()

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as ex:
        ns = parser.parse_args(args)

    captured = capsys.readouterr()
    assert "the following arguments are required" in captured.err
    err_desc = captured.err.split("the following arguments are required:")[-1]
    for arg in missing:
        assert arg in err_desc

    expected = ALL_ARGS - missing
    for exp in expected:
        assert exp not in err_desc


def test_parser():
    """Test that the parser succeeds when receiving expected args"""
    parser = get_parser()

    test_dir = "/foo/bar"
    test_freq = "123"

    cmd = f"-d {test_dir} -f {test_freq}"
    args = cmd.split()

    ns = parser.parse_args(args)

    assert ns.d == test_dir
    assert ns.f == test_freq


def test_ts():
    """Ensure expected output type"""
    ts = get_ts()
    assert isinstance(ts, int)


@pytest.mark.parametrize(
    ["etype", "name", "job_id", "step_id", "timestamp", "evt_type"],
    [
        pytest.param(
            "ensemble", "test-ensemble", "", "123", get_ts(), "start", id="start event"
        ),
        pytest.param(
            "ensemble", "test-ensemble", "", "123", get_ts(), "start", id="stop event"
        ),
    ],
)
def test_track_event(
    etype: str,
    name: str,
    job_id: str,
    step_id: str,
    timestamp: int,
    evt_type: str,
    fileutils,
):
    """Ensure that track event writes a file to the expected location"""
    exp_dir = fileutils.make_test_dir()
    persistable = PersistableEntity(etype, name, job_id, step_id, timestamp, exp_dir)

    exp_path = pathlib.Path(exp_dir)
    track_event(timestamp, persistable, evt_type, exp_path, logger)

    expected_output = exp_path / "manifest" / etype / name / f"{evt_type}.json"

    assert expected_output.exists()
    assert expected_output.is_file()


@pytest.mark.parametrize(
    ["evt_type", "track_fn"],
    [
        pytest.param("start", track_started, id="start event"),
        pytest.param("stop", track_completed, id="stop event"),
        pytest.param("step", track_timestep, id="update event"),
    ],
)
def test_track_specific(
    fileutils, evt_type: str, track_fn: t.Callable[[Job, logging.Logger], None]
):
    """Ensure that track start writes a file to the expected location with expected name"""

    etype = "ensemble"
    name = f"test-ensemble-{uuid.uuid4()}"
    job_id = ""
    step_id = "1234"
    timestamp = get_ts()

    exp_dir = fileutils.make_test_dir()
    persistable = PersistableEntity(etype, name, job_id, step_id, timestamp, exp_dir)

    exp_path = pathlib.Path(exp_dir)

    job = Job(name, job_id, persistable, "local", False)
    track_fn(job, logger)

    fname = f"{evt_type}.json"
    expected_output = exp_path / "manifest" / etype / name / fname

    assert expected_output.exists()
    assert expected_output.is_file()


def test_load_manifest(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly"""
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/telemetry.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    test_manifest_path = fileutils.make_test_file(
        "manifest.json", "manifest", sample_manifest.read_text()
    )
    test_manifest = pathlib.Path(test_manifest_path)
    assert test_manifest.exists()

    manifest = load_manifest(test_manifest_path)
    assert manifest.name == "my-experiment"
    assert str(manifest.path) == "experiment/path"
    assert manifest.launcher == "local"
    assert len(manifest.runs) == 1

    assert len(manifest.runs[0].models) == 2
    assert len(manifest.runs[0].orchestrators) == 1
    assert len(manifest.runs[0].ensembles) == 1


@pytest.mark.parametrize(
    ["job_id", "step_id", "etype", "exp_isorch", "exp_ismanaged"],
    [
        pytest.param("", "123", "model", False, False, id="unmanaged, non-orch"),
        pytest.param("456", "123", "ensemble", False, True, id="managed, non-orch"),
        pytest.param("789", "987", "orchestrator", True, True, id="managed, orch"),
        pytest.param("", "987", "orchestrator", True, False, id="unmanaged, orch"),
    ],
)
def test_persistable_computed_properties(
    job_id: str, step_id: str, etype: str, exp_isorch: bool, exp_ismanaged: bool
):
    name = f"test-{etype}-{uuid.uuid4()}"
    timestamp = get_ts()
    exp_dir = "/foo/bar"

    persistable = PersistableEntity(etype, name, job_id, step_id, timestamp, exp_dir)
    assert persistable.is_managed == exp_ismanaged
    assert persistable.is_orch == exp_isorch


@pytest.mark.parametrize(
    ["num_iters", "freq"],
    [
        pytest.param(1, 1, id="1 iter"),
        pytest.param(2, 1, id="2 iter"),
        pytest.param(3, 1, id="3 iter"),
    ],
)
def test_limit(fileutils, monkeypatch, capsys, num_iters: int, freq: int):
    """Verify the iteration limit is honored"""
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/telemetry.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    test_manifest_path = fileutils.make_test_file(
        "manifest.json", "manifest", sample_manifest.read_text()
    )
    test_manifest = pathlib.Path(test_manifest_path)
    assert test_manifest.exists()

    captured = capsys.readouterr()  # throw away existing output
    with monkeypatch.context() as ctx:
        ctx.setattr("smartsim.telemetrymonitor.ManifestEventHandler.on_timestep", lambda a,b,c: print("timestep!"))
        rc = main(freq, test_manifest.parent.parent, logger, num_iters=num_iters)

        captured = capsys.readouterr()
        m = re.findall(r"(timestep!)", captured.out)
        assert len(m) == num_iters

        models_dir = test_manifest.parent / "model"
        app1_dir = models_dir / "app1"
        app2_dir = models_dir / "app2"

        files = list(app1_dir.glob("*.json"))
        assert files

        files = list(app2_dir.glob("*.json"))
        assert files
        
        orchs_dir = test_manifest.parent / "orchestrator" / "orchestrator_0"
        files = list(orchs_dir.glob("*.json"))
        assert files
                
        ensemble_dir = test_manifest.parent / "ensemble" / "ensemble_1"
        files = list(ensemble_dir.glob("*.json"))
        assert files

    assert rc == 0
