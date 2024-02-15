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


import logging
import pathlib
import sys
import time
import typing as t
import uuid

import pytest

import smartsim._core.config.config as cfg
from conftest import FileUtils, WLMUtils
from smartsim import Experiment
from smartsim._core.control.job import Job, JobEntity
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.entrypoints.telemetrymonitor import (
    ManifestEventHandler,
    can_shutdown,
    event_loop,
    faux_return_code,
    get_parser,
    get_ts,
    hydrate_persistable,
    load_manifest,
    track_event,
)
from smartsim._core.launcher.launcher import WLMLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher
from smartsim._core.launcher.step.step import Step, proxyable_launch_cmd
from smartsim._core.launcher.stepInfo import StepInfo
from smartsim._core.utils import serialize
from smartsim.error.errors import UnproxyableStepError
from smartsim.settings.base import RunSettings
from smartsim.status import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEW,
    STATUS_PAUSED,
    STATUS_RUNNING,
)

ALL_ARGS = {"-exp_dir", "-frequency"}
PROXY_ENTRY_POINT = "smartsim._core.entrypoints.indirect"
CFG_TM_ENABLED_ATTR = "telemetry_enabled"


for_all_wlm_launchers = pytest.mark.parametrize(
    "wlm_launcher",
    [pytest.param(cls(), id=cls.__name__) for cls in WLMLauncher.__subclasses__()],
)

requires_wlm = pytest.mark.skipif(
    pytest.test_launcher == "local", reason="Test requires WLM"
)


logger = logging.getLogger()

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


@pytest.fixture(autouse=True)
def turn_on_tm(monkeypatch):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, property(lambda self: True))
    yield


def snooze_nonblocking(
    test_dir: pathlib.Path, max_delay: int = 20, post_data_delay: int = 2
):
    # let the non-blocking experiment complete.
    for _ in range(max_delay):
        time.sleep(1)
        if test_dir.exists():
            time.sleep(post_data_delay)
            break


@pytest.mark.parametrize(
    ["cmd", "missing"],
    [
        pytest.param("", {"-exp_dir", "-frequency"}, id="no args"),
        pytest.param("-exp_dir /foo/bar", {"-frequency"}, id="no freq"),
        pytest.param("-frequency 123", {"-exp_dir"}, id="no dir"),
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
    test_freq = 123

    cmd = f"-exp_dir {test_dir} -frequency {test_freq}"
    args = cmd.split()

    ns = parser.parse_args(args)

    assert ns.exp_dir == test_dir
    assert ns.frequency == test_freq


def test_ts():
    """Ensure expected output type"""
    ts = get_ts()
    assert isinstance(ts, int)


@pytest.mark.parametrize(
    ["etype", "task_id", "step_id", "timestamp", "evt_type"],
    [
        pytest.param("ensemble", "", "123", get_ts(), "start", id="start event"),
        pytest.param("ensemble", "", "123", get_ts(), "stop", id="stop event"),
    ],
)
def test_track_event(
    etype: str,
    task_id: str,
    step_id: str,
    timestamp: int,
    evt_type: str,
    test_dir: str,
):
    """Ensure that track event writes a file to the expected location"""
    exp_path = pathlib.Path(test_dir)
    track_event(timestamp, task_id, step_id, etype, evt_type, exp_path, logger)

    expected_output = exp_path / f"{evt_type}.json"

    assert expected_output.exists()
    assert expected_output.is_file()


def test_load_manifest(fileutils: FileUtils, test_dir: str, config: cfg.Config):
    """Ensure that the runtime manifest loads correctly"""
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/telemetry.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    test_manifest_path = fileutils.make_test_file(
        serialize.MANIFEST_FILENAME,
        pathlib.Path(test_dir) / config.telemetry_subdir,
        sample_manifest.read_text(),
    )
    test_manifest = pathlib.Path(test_manifest_path)
    assert test_manifest.exists()

    manifest = load_manifest(test_manifest_path)
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/path/to/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 6

    assert len(manifest.runs[0].models) == 1
    assert len(manifest.runs[2].models) == 8  # 8 models in ensemble
    assert len(manifest.runs[0].orchestrators) == 0
    assert len(manifest.runs[1].orchestrators) == 3  # 3 shards in db


def test_load_manifest_colo_model(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly when containing a colocated model"""
    # NOTE: for regeneration, this manifest can use `test_telemetry_colo`
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/colocatedmodel.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    manifest = load_manifest(sample_manifest_path)
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/tmp/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 1

    assert len(manifest.runs[0].models) == 1


def test_load_manifest_serial_models(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly when containing multiple models"""
    # NOTE: for regeneration, this manifest can use `test_telemetry_colo`
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/serialmodels.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    manifest = load_manifest(sample_manifest_path)
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/tmp/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 1

    assert len(manifest.runs[0].models) == 5


def test_load_manifest_db_and_models(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly when containing models &
    orchestrator across 2 separate runs"""
    # NOTE: for regeneration, this manifest can use `test_telemetry_colo`
    sample_manifest_path = fileutils.get_test_conf_path("telemetry/db_and_model.json")
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    manifest = load_manifest(sample_manifest_path)
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/tmp/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 2

    assert len(manifest.runs[0].orchestrators) == 1
    assert len(manifest.runs[1].models) == 1


def test_load_manifest_db_and_models_1run(fileutils: FileUtils):
    """Ensure that the runtime manifest loads correctly when containing models &
    orchestrator in a single run"""
    # NOTE: for regeneration, this manifest can use `test_telemetry_colo`
    sample_manifest_path = fileutils.get_test_conf_path(
        "telemetry/db_and_model_1run.json"
    )
    sample_manifest = pathlib.Path(sample_manifest_path)
    assert sample_manifest.exists()

    manifest = load_manifest(sample_manifest_path)
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/tmp/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 1

    assert len(manifest.runs[0].orchestrators) == 1
    assert len(manifest.runs[0].models) == 1


@pytest.mark.parametrize(
    ["task_id", "step_id", "etype", "exp_isorch", "exp_ismanaged"],
    [
        pytest.param("123", "", "model", False, False, id="unmanaged, non-orch"),
        pytest.param("456", "123", "ensemble", False, True, id="managed, non-orch"),
        pytest.param("789", "987", "orchestrator", True, True, id="managed, orch"),
        pytest.param("987", "", "orchestrator", True, False, id="unmanaged, orch"),
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


def test_shutdown_conditions():
    """Ensure conditions to shutdown telemetry monitor are correctly evaluated"""
    job_entity1 = JobEntity()
    job_entity1.name = "xyz"
    job_entity1.step_id = "123"
    job_entity1.task_id = ""

    logger = logging.getLogger()

    # show that an event handler w/no monitored jobs can shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    assert can_shutdown(mani_handler, logger)

    # show that an event handler w/a monitored job cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    mani_handler.job_manager.add_job(
        job_entity1.name, job_entity1.step_id, job_entity1, False
    )
    assert not can_shutdown(mani_handler, logger)
    assert not bool(mani_handler.job_manager.db_jobs)
    assert bool(mani_handler.job_manager.jobs)

    # show that an event handler w/a monitored db cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(
        job_entity1.name, job_entity1.step_id, job_entity1, False
    )
    assert not can_shutdown(mani_handler, logger)
    assert bool(mani_handler.job_manager.db_jobs)
    assert not bool(mani_handler.job_manager.jobs)

    # show that an event handler w/a dbs & tasks cannot shutdown
    job_entity2 = JobEntity()
    job_entity2.name = "xyz"
    job_entity2.step_id = "123"
    job_entity2.task_id = ""

    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(
        job_entity1.name, job_entity1.step_id, job_entity1, False
    )

    mani_handler.job_manager.add_job(
        job_entity2.name, job_entity2.step_id, job_entity2, False
    )
    assert not can_shutdown(mani_handler, logger)
    assert bool(mani_handler.job_manager.db_jobs)
    assert bool(mani_handler.job_manager.jobs)

    # ... now, show that removing 1 of 2 jobs still doesn't shutdown
    mani_handler.job_manager.db_jobs.popitem()
    assert not can_shutdown(mani_handler, logger)

    # ... now, show that removing final job will allow shutdown
    mani_handler.job_manager.jobs.popitem()
    assert can_shutdown(mani_handler, logger)


def test_auto_shutdown():
    """Ensure that the cooldown timer is respected"""

    class FauxObserver:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

        def is_alive(self) -> bool:
            if self.stop_count > 0:
                return False

            return True

    job_entity1 = JobEntity()
    job_entity1.name = "xyz"
    job_entity1.step_id = "123"
    job_entity1.task_id = ""

    frequency = 1

    # show that an event handler w/out a monitored task will automatically stop
    mani_handler = ManifestEventHandler("xyz", logger)
    observer = FauxObserver()
    duration = 2

    ts0 = get_ts()
    event_loop(observer, mani_handler, frequency, logger, cooldown_duration=duration)
    ts1 = get_ts()

    assert ts1 - ts0 >= duration
    assert observer.stop_count == 1

    # show that the new cooldown duration is respected
    mani_handler = ManifestEventHandler("xyz", logger)
    observer = FauxObserver()
    duration = 5

    ts0 = get_ts()
    event_loop(observer, mani_handler, frequency, logger, cooldown_duration=duration)
    ts1 = get_ts()

    assert ts1 - ts0 >= duration
    assert observer.stop_count == 1


def test_telemetry_single_model(fileutils, test_dir, wlmutils, config):
    """Test that it is possible to create_database then colocate_db_uds/colocate_db_tcp
    with unique db_identifiers"""

    # Set experiment name
    exp_name = "telemetry_single_model"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_script = fileutils.get_test_conf_path("echo.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create run settings
    app_settings = exp.create_run_settings(sys.executable, test_script)
    app_settings.set_nodes(1)
    app_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("perroquet", app_settings)
    exp.generate(smartsim_model)
    exp.start(smartsim_model, block=True)
    assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

    telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
    start_events = list(telemetry_output_path.rglob("start.json"))
    stop_events = list(telemetry_output_path.rglob("stop.json"))

    assert len(start_events) == 1
    assert len(stop_events) == 1


def test_telemetry_single_model_nonblocking(
    fileutils, test_dir, wlmutils, monkeypatch, config
):
    """Ensure that the telemetry monitor logs exist when the experiment
    is non-blocking"""
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "test_telemetry_single_model_nonblocking"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create run settings
        app_settings = exp.create_run_settings(sys.executable, test_script)
        app_settings.set_nodes(1)
        app_settings.set_tasks_per_node(1)

        # Create the SmartSim Model
        smartsim_model = exp.create_model("perroquet", app_settings)
        exp.generate(smartsim_model)
        exp.start(smartsim_model)

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=30)

        assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        assert len(start_events) == 1
        assert len(stop_events) == 1


def test_telemetry_serial_models(fileutils, test_dir, wlmutils, monkeypatch, config):
    """
    Test telemetry with models being run in serial (one after each other)
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_serial_models"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create run settings
        app_settings = exp.create_run_settings(sys.executable, test_script)
        app_settings.set_nodes(1)
        app_settings.set_tasks_per_node(1)

        # Create the SmartSim Model
        smartsim_models = [
            exp.create_model(f"perroquet_{i}", app_settings) for i in range(5)
        ]
        exp.generate(*smartsim_models)
        exp.start(*smartsim_models, block=True)
        assert all(
            [status == STATUS_COMPLETED for status in exp.get_status(*smartsim_models)]
        )

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        assert len(start_events) == 5
        assert len(stop_events) == 5


def test_telemetry_serial_models_nonblocking(
    fileutils, test_dir, wlmutils, monkeypatch, config
):
    """
    Test telemetry with models being run in serial (one after each other)
    in a non-blocking experiment
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_serial_models"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create run settings
        app_settings = exp.create_run_settings(sys.executable, test_script)
        app_settings.set_nodes(1)
        app_settings.set_tasks_per_node(1)

        # Create the SmartSim Model
        smartsim_models = [
            exp.create_model(f"perroquet_{i}", app_settings) for i in range(5)
        ]
        exp.generate(*smartsim_models)
        exp.start(*smartsim_models)

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=10)

        assert all(
            [status == STATUS_COMPLETED for status in exp.get_status(*smartsim_models)]
        )

        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        assert len(start_events) == 5
        assert len(stop_events) == 5


def test_telemetry_db_only_with_generate(test_dir, wlmutils, monkeypatch, config):
    """
    Test telemetry with only a database running
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_db_with_generate"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_interface = wlmutils.get_test_interface()
        test_port = wlmutils.get_test_port()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        exp.generate(orc)

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir

        try:
            exp.start(orc, block=True)

            snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=10)

            start_events = list(telemetry_output_path.rglob("start.json"))
            stop_events = list(telemetry_output_path.rglob("stop.json"))

            assert len(start_events) == 1
            assert len(stop_events) <= 1
        finally:
            exp.stop(orc)
            snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=10)

        assert exp.get_status(orc)[0] == STATUS_CANCELLED

        stop_events = list(telemetry_output_path.rglob("stop.json"))
        assert len(stop_events) == 1


def test_telemetry_db_only_without_generate(test_dir, wlmutils, monkeypatch, config):
    """
    Test telemetry with only a non-generated database running
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_db_only_without_generate"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_interface = wlmutils.get_test_interface()
        test_port = wlmutils.get_test_port()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        orc.set_path(test_dir)
        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir

        try:
            exp.start(orc)

            snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=30)

            start_events = list(telemetry_output_path.rglob("start.json"))
            stop_events = list(telemetry_output_path.rglob("stop.json"))

            assert len(start_events) == 1
            assert len(stop_events) == 0
        finally:
            exp.stop(orc)

        snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=10)
        assert exp.get_status(orc)[0] == STATUS_CANCELLED

        stop_events = list(telemetry_output_path.rglob("stop.json"))
        assert len(stop_events) == 1


def test_telemetry_db_and_model(fileutils, test_dir, wlmutils, monkeypatch, config):
    """
    Test telemetry with only a database and a model running
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_db_and_model"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_interface = wlmutils.get_test_interface()
        test_port = wlmutils.get_test_port()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        exp.generate(orc)
        try:
            exp.start(orc)

            # create run settings
            app_settings = exp.create_run_settings(sys.executable, test_script)
            app_settings.set_nodes(1)
            app_settings.set_tasks_per_node(1)

            # Create the SmartSim Model
            smartsim_model = exp.create_model("perroquet", app_settings)
            exp.generate(smartsim_model)
            exp.start(smartsim_model, block=True)
        finally:
            exp.stop(orc)

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=30)

        assert exp.get_status(orc)[0] == STATUS_CANCELLED
        assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

        start_events = list(telemetry_output_path.rglob("database/**/start.json"))
        stop_events = list(telemetry_output_path.rglob("database/**/stop.json"))

        assert len(start_events) == 1
        assert len(stop_events) == 1

        start_events = list(telemetry_output_path.rglob("model/**/start.json"))
        stop_events = list(telemetry_output_path.rglob("model/**/stop.json"))
        assert len(start_events) == 1
        assert len(stop_events) == 1


def test_telemetry_ensemble(fileutils, test_dir, wlmutils, monkeypatch, config):
    """
    Test telemetry with only an ensemble
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_ensemble"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        app_settings = exp.create_run_settings(sys.executable, test_script)
        app_settings.set_nodes(1)
        app_settings.set_tasks_per_node(1)

        ens = exp.create_ensemble("troupeau", run_settings=app_settings, replicas=5)
        exp.generate(ens)
        exp.start(ens, block=True)
        assert all([status == STATUS_COMPLETED for status in exp.get_status(ens)])

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        snooze_nonblocking(telemetry_output_path, max_delay=60, post_data_delay=30)
        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        assert len(start_events) == 5
        assert len(stop_events) == 5


def test_telemetry_colo(fileutils, test_dir, wlmutils, coloutils, monkeypatch, config):
    """
    Test telemetry with only a colocated model running
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_colo"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        smartsim_model = coloutils.setup_test_colo(
            fileutils,
            "uds",
            exp,
            "echo.py",
            {},
        )

        exp.generate(smartsim_model)
        exp.start(smartsim_model, block=True)
        assert all(
            [status == STATUS_COMPLETED for status in exp.get_status(smartsim_model)]
        )

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        # the colodb does NOT show up as a unique entity in the telemetry
        assert len(start_events) == 1
        assert len(stop_events) == 1


@pytest.mark.parametrize(
    "frequency, cooldown",
    [
        pytest.param(1, 1, id="1s shutdown"),
        pytest.param(1, 5, id="5s shutdown"),
        pytest.param(1, 15, id="15s shutdown"),
    ],
)
def test_telemetry_autoshutdown(
    test_dir, wlmutils, monkeypatch, frequency, cooldown, config
):
    """
    Ensure that the telemetry monitor process shuts down after the desired
    cooldown period
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", frequency)
        ctx.setattr(cfg.Config, "telemetry_cooldown", cooldown)

        # Set experiment name
        exp_name = "telemetry_ensemble"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        start_time = get_ts()
        stop_time = start_time
        exp.start(block=False)

        telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
        empty_mani = list(telemetry_output_path.rglob("manifest.json"))
        assert len(empty_mani) == 1, "an  manifest.json should be created"

        popen = exp._control._telemetry_monitor
        assert popen.pid > 0
        assert popen.returncode is None

        # give some leeway during testing for the cooldown to get hit
        for i in range(10):
            if popen.poll() is not None:
                stop_time = get_ts()
                print(f"Completed polling for telemetry shutdown after {i} attempts")
                break
            time.sleep(3)

        assert popen.returncode is not None
        assert stop_time >= (start_time + cooldown)


class MockStep(Step):
    """Mock step to implement any abstract methods so that it can be
    instanced for test purposes
    """

    def get_launch_cmd(self):
        return ["spam", "eggs"]


@pytest.fixture
def mock_step_meta_dict(test_dir, config):
    telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir
    yield {
        "entity_type": "mock",
        "status_dir": telemetry_output_path,
    }


@pytest.fixture
def mock_step(test_dir, mock_step_meta_dict):
    rs = RunSettings("echo")
    step = MockStep("mock-step", test_dir, rs)
    step.meta = mock_step_meta_dict
    yield step


def test_proxy_launch_cmd_decorator_reformats_cmds(mock_step, monkeypatch):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, True)
    get_launch_cmd = proxyable_launch_cmd(lambda step: ["some", "cmd", "list"])
    cmd = get_launch_cmd(mock_step)
    assert cmd != ["some", "cmd", "list"]
    assert sys.executable in cmd
    assert PROXY_ENTRY_POINT in cmd


def test_proxy_launch_cmd_decorator_does_not_reformat_cmds_if_the_tm_is_off(
    mock_step, monkeypatch
):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, False)
    get_launch_cmd = proxyable_launch_cmd(lambda step: ["some", "cmd", "list"])
    cmd = get_launch_cmd(mock_step)
    assert cmd == ["some", "cmd", "list"]


def test_proxy_launch_cmd_decorator_errors_if_attempt_to_proxy_a_managed_step(
    mock_step, monkeypatch
):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, True)
    mock_step.managed = True
    get_launch_cmd = proxyable_launch_cmd(lambda step: ["some", "cmd", "list"])
    with pytest.raises(UnproxyableStepError):
        get_launch_cmd(mock_step)


@for_all_wlm_launchers
def test_unmanaged_steps_are_proxyed_through_indirect(
    wlm_launcher, mock_step_meta_dict, test_dir, monkeypatch
):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, True)
    rs = RunSettings("echo", ["hello", "world"])
    step = wlm_launcher.create_step("test-step", test_dir, rs)
    step.meta = mock_step_meta_dict
    assert isinstance(step, Step)
    assert not step.managed
    cmd = step.get_launch_cmd()
    assert sys.executable in cmd
    assert PROXY_ENTRY_POINT in cmd
    assert "hello" not in cmd
    assert "world" not in cmd


@for_all_wlm_launchers
def test_unmanaged_steps_are_not_proxied_if_the_telemetry_monitor_is_disabled(
    wlm_launcher, mock_step_meta_dict, test_dir, monkeypatch
):
    monkeypatch.setattr(cfg.Config, CFG_TM_ENABLED_ATTR, False)
    rs = RunSettings("echo", ["hello", "world"])
    step = wlm_launcher.create_step("test-step", test_dir, rs)
    step.meta = mock_step_meta_dict
    assert isinstance(step, Step)
    assert not step.managed
    cmd = step.get_launch_cmd()
    assert PROXY_ENTRY_POINT not in cmd
    assert "hello" in cmd
    assert "world" in cmd


@requires_wlm
@pytest.mark.parametrize(
    "run_command",
    [
        pytest.param("", id="Unmanaged"),
        pytest.param("auto", id="Managed"),
    ],
)
def test_multistart_experiment(
    wlmutils: WLMUtils,
    fileutils: FileUtils,
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    run_command: str,
    config: cfg.Config,
):
    """Run an experiment with multiple start calls to ensure that telemetry is
    saved correctly for each run
    """

    exp_name = "my-exp"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    rs_e = exp.create_run_settings(
        sys.executable, ["printing_model.py"], run_command=run_command
    )
    rs_e.set_nodes(1)
    rs_e.set_tasks(1)
    ens = exp.create_ensemble(
        "my-ens",
        run_settings=rs_e,
        perm_strategy="all_perm",
        params={
            "START": ["spam"],
            "MID": ["eggs"],
            "END": ["sausage", "and spam"],
        },
    )

    test_script_path = fileutils.get_test_conf_path("printing_model.py")
    ens.attach_generator_files(to_configure=[test_script_path])

    rs_m = exp.create_run_settings("echo", ["hello", "world"], run_command=run_command)
    rs_m.set_nodes(1)
    rs_m.set_tasks(1)
    model = exp.create_model("my-model", run_settings=rs_m)

    db = exp.create_database(
        db_nodes=1,
        port=wlmutils.get_test_port(),
        interface=wlmutils.get_test_interface(),
    )

    exp.generate(db, ens, model, overwrite=True)

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)
        ctx.setattr(cfg.Config, "telemetry_cooldown", 45)

        exp.start(model, block=False)

        # track PID to see that telmon cooldown avoids restarting process
        tm_pid = exp._control._telemetry_monitor.pid

        exp.start(db, block=False)
        # check that same TM proc is active
        assert tm_pid == exp._control._telemetry_monitor.pid
        try:
            exp.start(ens, block=True, summary=True)
        finally:
            exp.stop(db)
            assert tm_pid == exp._control._telemetry_monitor.pid
            time.sleep(3)  # time for telmon to write db stop event

    telemetry_output_path = pathlib.Path(test_dir) / config.telemetry_subdir

    db_start_events = list(telemetry_output_path.rglob("database/**/start.json"))
    db_stop_events = list(telemetry_output_path.rglob("database/**/stop.json"))
    assert len(db_start_events) == 1
    assert len(db_stop_events) == 1

    m_start_events = list(telemetry_output_path.rglob("model/**/start.json"))
    m_stop_events = list(telemetry_output_path.rglob("model/**/stop.json"))
    assert len(m_start_events) == 1
    assert len(m_stop_events) == 1

    e_start_events = list(telemetry_output_path.rglob("ensemble/**/start.json"))
    e_stop_events = list(telemetry_output_path.rglob("ensemble/**/stop.json"))
    assert len(e_start_events) == 2
    assert len(e_stop_events) == 2


@pytest.mark.parametrize(
    "status_in, expected_out",
    [
        pytest.param(STATUS_CANCELLED, 1, id="failure on cancellation"),
        pytest.param(STATUS_COMPLETED, 0, id="success on completion"),
        pytest.param(STATUS_FAILED, 1, id="failure on failed"),
        pytest.param(STATUS_NEW, None, id="failure on new"),
        pytest.param(STATUS_PAUSED, None, id="failure on paused"),
        pytest.param(STATUS_RUNNING, None, id="failure on running"),
    ],
)
def test_faux_rc(status_in: str, expected_out: t.Optional[int]):
    """Ensure faux response codes match expectations."""
    step_info = StepInfo(status=status_in)

    rc = faux_return_code(step_info)
    assert rc == expected_out


@pytest.mark.parametrize(
    "status_in, expected_out, expected_has_jobs",
    [
        pytest.param(STATUS_CANCELLED, 1, False, id="failure on cancellation"),
        pytest.param(STATUS_COMPLETED, 0, False, id="success on completion"),
        pytest.param(STATUS_FAILED, 1, False, id="failure on failed"),
        pytest.param(STATUS_NEW, None, True, id="failure on new"),
        pytest.param(STATUS_PAUSED, None, True, id="failure on paused"),
        pytest.param(STATUS_RUNNING, None, True, id="failure on running"),
    ],
)
def test_wlm_completion_handling(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    status_in: str,
    expected_out: t.Optional[int],
    expected_has_jobs: bool,
):
    def get_faux_update(status: str) -> t.Callable:
        def _faux_updates(_self: WLMLauncher, _names: t.List[str]) -> t.List[StepInfo]:
            return [("faux-name", StepInfo(status=status))]

        return _faux_updates

    ts = get_ts()
    with monkeypatch.context() as ctx:
        # don't actually start a job manager
        ctx.setattr(JobManager, "start", lambda x: ...)
        ctx.setattr(SlurmLauncher, "get_step_update", get_faux_update(status_in))

        mani_handler = ManifestEventHandler("xyz", logger)
        mani_handler.set_launcher("slurm")

        # prep a fake job to request updates for
        job_entity = JobEntity()
        job_entity.name = "faux-name"
        job_entity.step_id = "faux-step-id"
        job_entity.task_id = 1234
        job_entity.status_dir = test_dir
        job_entity.type = "orchestrator"

        job = Job(job_entity.name, job_entity.step_id, job_entity, "slurm", True)

        # populate our tracking collections
        mani_handler._tracked_jobs = {job_entity.key: job_entity}
        mani_handler.job_manager.jobs[job.name] = job

        mani_handler.on_timestep(ts)

        # see that the job queue was properly manipulated
        has_jobs = bool(mani_handler._tracked_jobs)
        assert expected_has_jobs == has_jobs

        # see that the event was properly written
        stop_event_path = pathlib.Path(test_dir) / "stop.json"

        # if a status wasn't terminal, no stop event should have been written
        should_have_stop_event = False if expected_out is None else True
        assert should_have_stop_event == stop_event_path.exists()
