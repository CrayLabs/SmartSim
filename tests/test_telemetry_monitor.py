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
from random import sample
import pytest
import re
import typing as t
import uuid
from conftest import FileUtils
from smartsim._core.control.job import Job

from smartsim._core.entrypoints.telemetrymonitor import (
    get_parser,
    get_ts,
    main,
    track_event,
    track_started,
    track_completed,
    track_timestep,
    load_manifest,
    hydrate_persistable,
)
from smartsim._core.utils import serialize


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
    ["etype", "task_id", "step_id", "timestamp", "evt_type"],
    [
        pytest.param(
            "ensemble", "", "123", get_ts(), "start", id="start event"
        ),
        pytest.param(
            "ensemble", "", "123", get_ts(), "start", id="stop event"
        ),
    ],
)
def test_track_event(
    etype: str,
    task_id: str,
    step_id: str,
    timestamp: int,
    evt_type: str,
    fileutils,
):
    """Ensure that track event writes a file to the expected location"""
    exp_dir = fileutils.make_test_dir()
    exp_path = pathlib.Path(exp_dir)
    track_event(timestamp, task_id, step_id, etype, evt_type, exp_path, logger)

    expected_output = exp_path / f"{evt_type}.json"

    assert expected_output.exists()
    assert expected_output.is_file()


@pytest.mark.parametrize(
    ["evt_type", "track_fn"],
    [
        pytest.param("start", track_started, id="start event"),
        pytest.param("stop", track_completed, id="stop event"),
        pytest.param("timestep", track_timestep, id="update event"),
    ],
)
def test_track_specific(
    fileutils, evt_type: str, track_fn: t.Callable[[Job, logging.Logger], None]
):
    """Ensure that track start writes a file to the expected location with expected name"""

    etype = "ensemble"
    name = f"test-ensemble-{uuid.uuid4()}"
    task_id = ""
    step_id = "1234"
    timestamp = get_ts()

    exp_dir = pathlib.Path(fileutils.make_test_dir())
    stored = {
        "name": name,
        "run_id": timestamp,
        "telemetry_metadata": {
            "status_dir": str(exp_dir / serialize.TELMON_SUBDIR),
            "task_id": task_id,
            "step_id": step_id,        
        },
    }
    persistables = hydrate_persistable(etype, stored, exp_dir)
    persistable = persistables[0] if persistables else None

    job = Job(name, task_id, persistable, "local", False)
    
    track_fn(job, logger)

    fname = f"{evt_type}.json"
    expected_output = exp_dir / serialize.TELMON_SUBDIR / fname

    assert expected_output.exists()
    assert expected_output.is_file()


def test_load_manifest(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly"""
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/telemetry.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    test_manifest_path = fileutils.make_test_file(
        serialize.MANIFEST_FILENAME, serialize.TELMON_SUBDIR, sample_manifest.read_text()
    )
    test_manifest = pathlib.Path(test_manifest_path)
    assert test_manifest.exists()

    manifest = load_manifest(test_manifest_path)
    assert manifest.name == "my-experiment"
    assert str(manifest.path) == "experiment/path"
    assert manifest.launcher == "local"
    assert len(manifest.runs) == 1

    assert len(manifest.runs[0].models) == 2
    assert len(manifest.runs[0].orchestrators) == 2
    assert len(manifest.runs[0].ensembles) == 1


@pytest.mark.parametrize(
    ["task_id", "step_id", "etype", "exp_isorch", "exp_ismanaged"],
    [
        pytest.param("", "123", "model", False, False, id="unmanaged, non-orch"),
        pytest.param("456", "123", "ensemble", False, True, id="managed, non-orch"),
        pytest.param("789", "987", "orchestrator", True, True, id="managed, orch"),
        pytest.param("", "987", "orchestrator", True, False, id="unmanaged, orch"),
    ],
)
def test_persistable_computed_properties(
    task_id: str, step_id: str, etype: str, exp_isorch: bool, exp_ismanaged: bool
):
    name = f"test-{etype}-{uuid.uuid4()}"
    timestamp = get_ts()
    exp_dir = pathlib.Path("/foo/bar")
    stored = {
        "name": name,
        "run_id": timestamp,
        "telemetry_metadata": {
            "status_dir": str(exp_dir),
            "task_id": task_id,
            "step_id": step_id,
        },
    }
    persistables = hydrate_persistable(etype, stored, exp_dir)
    persistable = persistables[0] if persistables else None

    assert persistable.is_managed == exp_ismanaged
    assert persistable.is_db == exp_isorch


def test_deserialize_ensemble(fileutils: FileUtils):
    """Ensure that the children of ensembles (models) are correctly 
    placed in the models collection"""
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/ensembles.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    manifest = load_manifest(sample_manifest_path)
    assert manifest

    assert len(manifest.runs) == 1

    # NOTE: no longer returning ensembles, only children...
    # assert len(manifest.runs[0].ensembles) == 1  
    assert len(manifest.runs[0].models) == 8
