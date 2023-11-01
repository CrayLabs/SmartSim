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
import typing as t
import time
import uuid
from conftest import FileUtils
import smartsim._core.config.config as cfg
from smartsim._core.control.job import Job, JobEntity
from smartsim.status import STATUS_COMPLETED, STATUS_CANCELLED


from smartsim._core.entrypoints.telemetrymonitor import (
    can_shutdown,
    get_parser,
    get_ts,
    shutdown_when_completed,
    track_event,
    track_timestep,
    load_manifest,
    hydrate_persistable,
    ManifestEventHandler,
)
from smartsim._core.utils import serialize
from smartsim import Experiment


ALL_ARGS = {"-d", "-f"}
logger = logging.getLogger()


def snooze_nonblocking(test_dir: str, max_delay: int = 20, post_data_delay: int = 2):
    telmon_subdir = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
    # let the non-blocking experiment complete.
    for _ in range(max_delay):
        time.sleep(1)
        if telmon_subdir.exists():
            time.sleep(post_data_delay)
            break


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
    persistable = persistables[etype][0] if persistables else None

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
    assert manifest.name == "my-exp"
    assert str(manifest.path) == "/lus/cls01029/drozt/playground/ss/dash-int/my-exp"
    assert manifest.launcher == "Slurm"
    assert len(manifest.runs) == 6

    assert len(manifest.runs[0].models) == 1
    assert len(manifest.runs[2].models) == 8  # 8 models in ensemble
    assert len(manifest.runs[0].orchestrators) == 0
    assert len(manifest.runs[1].orchestrators) == 3  # 3 shards in db
    # assert len(manifest.runs[0].ensembles) == 1


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
    persistable = persistables[etype][0] if persistables[etype] else None

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
    job_entity1.job_id = "123"
    job_entity1.step_id = ""

    # show that an event handler w/no monitored jobs can shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    assert can_shutdown(mani_handler)

    # show that an event handler w/a monitored job cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)
    assert not can_shutdown(mani_handler)
    assert not bool(mani_handler.job_manager.db_jobs)
    assert bool(mani_handler.job_manager.jobs)

    # show that an event handler w/a monitored db cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)
    assert not can_shutdown(mani_handler)
    assert bool(mani_handler.job_manager.db_jobs)
    assert not bool(mani_handler.job_manager.jobs)

    # show that an event handler w/a dbs & tasks cannot shutdown
    job_entity2 = JobEntity()
    job_entity2.name = "xyz"
    job_entity2.job_id = "123"
    job_entity2.step_id = ""

    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)

    mani_handler.job_manager.add_job(job_entity2.name,
                                    job_entity2.job_id,
                                    job_entity2,
                                    False)
    assert not can_shutdown(mani_handler)
    assert bool(mani_handler.job_manager.db_jobs)
    assert bool(mani_handler.job_manager.jobs)

    # ... now, show that removing 1 of 2 jobs still doesn't shutdown
    mani_handler.job_manager.db_jobs.popitem()
    assert not can_shutdown(mani_handler)

    # ... now, show that removing final job will allow shutdown
    mani_handler.job_manager.jobs.popitem()
    assert can_shutdown(mani_handler)


def test_shutdown_action():
    """Ensure file system listener is properly shutdown"""
    class FauxObserver:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    job_entity1 = JobEntity()
    job_entity1.name = "xyz"
    job_entity1.job_id = "123"
    job_entity1.step_id = ""

    # show that an event handler w/no monitored jobs can shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 1

    # show that an event handler w/a monitored job cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 0

    # show that an event handler w/a monitored db cannot shutdown
    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 0

    # show that an event handler w/a dbs & tasks cannot shutdown
    job_entity2 = JobEntity()
    job_entity2.name = "xyz"
    job_entity2.job_id = "123"
    job_entity2.step_id = ""

    mani_handler = ManifestEventHandler("xyz", logger)
    job_entity1.type = "orchestrator"
    mani_handler.job_manager.add_job(job_entity1.name,
                                     job_entity1.job_id,
                                     job_entity1,
                                     False)

    mani_handler.job_manager.add_job(job_entity2.name,
                                    job_entity2.job_id,
                                    job_entity2,
                                    False)
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 0

    # ... now, show that removing 1 of 2 jobs still doesn't shutdown
    mani_handler.job_manager.db_jobs.popitem()
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 0

    # ... now, show that removing final job will allow shutdown
    mani_handler.job_manager.jobs.popitem()
    observer = FauxObserver()
    shutdown_when_completed(observer, mani_handler)
    assert observer.stop_count == 1


def test_telemetry_single_model(fileutils, wlmutils):
    """Test that it is possible to create_database then colocate_db_uds/colocate_db_tcp
    with unique db_identifiers"""

    # Set experiment name
    exp_name = "telemetry_single_model"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("echo.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create run settings
    app_settings = exp.create_run_settings("python", test_script)
    app_settings.set_nodes(1)
    app_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_model = exp.create_model("perroquet", app_settings)
    exp.generate(smartsim_model)
    exp.start(smartsim_model, block=True)
    assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

    telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
    start_events = list(telemetry_output_path.rglob("start.json"))
    stop_events = list(telemetry_output_path.rglob("stop.json"))

    assert len(start_events) == 1
    assert len(stop_events) == 1


def test_telemetry_single_model_nonblocking(fileutils, wlmutils):
    """Ensure that the telemetry monitor logs exist when the experiment 
    is non-blocking"""

    # Set experiment name
    exp_name = "test_telemetry_single_model_nonblocking"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("echo.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create run settings
    app_settings = exp.create_run_settings("python", test_script)
    app_settings.set_nodes(1)
    app_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_model = exp.create_model("perroquet", app_settings)
    exp.generate(smartsim_model)
    exp.start(smartsim_model)

    snooze_nonblocking(test_dir)

    assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

    telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
    start_events = list(telemetry_output_path.rglob("start.json"))
    stop_events = list(telemetry_output_path.rglob("stop.json"))

    assert len(start_events) == 1
    assert len(stop_events) == 1


def test_telemetry_serial_models(fileutils, wlmutils):
    """
    Test telemetry with models being run in serial (one after each other)
    """

    # Set experiment name
    exp_name = "telemetry_serial_models"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("echo.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create run settings
    app_settings = exp.create_run_settings("python", test_script)
    app_settings.set_nodes(1)
    app_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_models = [ exp.create_model(f"perroquet_{i}", app_settings) for i in range(5) ]
    exp.generate(*smartsim_models)
    exp.start(*smartsim_models, block=True)
    assert all([status == STATUS_COMPLETED for status in exp.get_status(*smartsim_models)])

    telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
    start_events = list(telemetry_output_path.rglob("start.json"))
    stop_events = list(telemetry_output_path.rglob("stop.json"))

    assert len(start_events) == 5
    assert len(stop_events) == 5


def test_telemetry_serial_models_nonblocking(fileutils, wlmutils):
    """
    Test telemetry with models being run in serial (one after each other)
    in a non-blocking experiment
    """

    # Set experiment name
    exp_name = "telemetry_serial_models"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("echo.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create run settings
    app_settings = exp.create_run_settings("python", test_script)
    app_settings.set_nodes(1)
    app_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_models = [ exp.create_model(f"perroquet_{i}", app_settings) for i in range(5) ]
    exp.generate(*smartsim_models)
    exp.start(*smartsim_models)

    snooze_nonblocking(test_dir, max_delay=45, post_data_delay=10)

    assert all([status == STATUS_COMPLETED for status in exp.get_status(*smartsim_models)])

    telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
    start_events = list(telemetry_output_path.rglob("start.json"))
    stop_events = list(telemetry_output_path.rglob("stop.json"))

    assert len(start_events) == 5
    assert len(stop_events) == 5


def test_telemetry_db_only_with_generate(fileutils, wlmutils, monkeypatch):
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
        test_dir = fileutils.make_test_dir()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        exp.generate(orc)
        try:
            exp.start(orc, block=True)

            telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
            start_events = list(telemetry_output_path.rglob("start.json"))
            stop_events = list(telemetry_output_path.rglob("stop.json"))

            assert len(start_events) == 1
            assert len(stop_events) == 0
        finally:
            exp.stop(orc)

        time.sleep(3)
        assert exp.get_status(orc)[0] == STATUS_CANCELLED

        stop_events = list(telemetry_output_path.rglob("stop.json"))
        assert len(stop_events) == 1


def test_telemetry_db_only_without_generate(fileutils, wlmutils, monkeypatch):
    """
    Test telemetry with only a database running
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_db_only_without_generate"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_interface = wlmutils.get_test_interface()
        test_port = wlmutils.get_test_port()
        test_dir = fileutils.make_test_dir()

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        try:
            exp.start(orc)

            telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
            start_events = list(telemetry_output_path.rglob("start.json"))
            stop_events = list(telemetry_output_path.rglob("stop.json"))

            assert len(start_events) == 1
            assert len(stop_events) == 0
        finally:
            exp.stop(orc)
        
        time.sleep(3)
        assert exp.get_status(orc)[0] == STATUS_CANCELLED

        stop_events = list(telemetry_output_path.rglob("stop.json"))
        assert len(stop_events) == 1


def test_telemetry_db_and_model(fileutils, wlmutils, monkeypatch):
    """
    Test telemetry with only a database running
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_db_and_model"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_interface = wlmutils.get_test_interface()
        test_port = wlmutils.get_test_port()
        test_dir = fileutils.make_test_dir()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        # create regular database
        orc = exp.create_database(port=test_port, interface=test_interface)
        try:
            exp.start(orc)
            # create run settings
            app_settings = exp.create_run_settings("python", test_script)
            app_settings.set_nodes(1)
            app_settings.set_tasks_per_node(1)

            # Create the SmartSim Model
            smartsim_model = exp.create_model("perroquet", app_settings)
            exp.generate(smartsim_model)
            exp.start(smartsim_model, block=True)
        finally:
            exp.stop(orc)
            time.sleep(3)

        assert exp.get_status(orc)[0] == STATUS_CANCELLED
        assert exp.get_status(smartsim_model)[0] == STATUS_COMPLETED

        telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR

        start_events = list(telemetry_output_path.rglob("dbnode/*/start.json"))
        stop_events = list(telemetry_output_path.rglob("dbnode/*/stop.json"))

        assert len(start_events) == 1
        assert len(stop_events) == 1

        start_events = list(telemetry_output_path.rglob("model/*/start.json"))
        stop_events = list(telemetry_output_path.rglob("model/*/stop.json"))
        assert len(start_events) == 1
        assert len(stop_events) == 1


def test_telemetry_ensemble(fileutils, wlmutils, monkeypatch):
    """
    Test telemetry with only a database running
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_ensemble"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_dir = fileutils.make_test_dir()
        test_script = fileutils.get_test_conf_path("echo.py")

        # Create SmartSim Experiment
        exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

        app_settings = exp.create_run_settings("python", test_script)
        app_settings.set_nodes(1)
        app_settings.set_tasks_per_node(1)

        ens = exp.create_ensemble("troupeau", run_settings=app_settings, replicas=5)
        exp.generate(ens)
        exp.start(ens, block=True)
        assert all([status == STATUS_COMPLETED for status in exp.get_status(ens)])

        time.sleep(3)
        telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        assert len(start_events) == 5
        assert len(stop_events) == 5


def test_telemetry_colo(fileutils, wlmutils, coloutils, monkeypatch):
    """
    Test telemetry with only a database running
    """

    with monkeypatch.context() as ctx:
        ctx.setattr(cfg.Config, "telemetry_frequency", 1)

        # Set experiment name
        exp_name = "telemetry_ensemble"

        # Retrieve parameters from testing environment
        test_launcher = wlmutils.get_test_launcher()
        test_dir = fileutils.make_test_dir()

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
        assert all([status == STATUS_COMPLETED for status in exp.get_status(smartsim_model)])

        time.sleep(3)
        telemetry_output_path = pathlib.Path(test_dir) / serialize.TELMON_SUBDIR
        start_events = list(telemetry_output_path.rglob("start.json"))
        stop_events = list(telemetry_output_path.rglob("stop.json"))

        # the colodb does NOT show up as a unique entity in the telemetry
        assert len(start_events) == 1
        assert len(stop_events) == 1
