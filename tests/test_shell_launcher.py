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

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest.mock

import psutil
import pytest

from smartsim._core.shell.shell_launcher import ShellLauncher, ShellLauncherCommand, sp
from smartsim._core.utils import helpers
from smartsim._core.utils.shell import *
from smartsim.entity import entity
from smartsim.error.errors import LauncherJobNotFound
from smartsim.status import JobStatus

pytestmark = pytest.mark.group_a


class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity"""

    def __init__(self):
        super().__init__("test-entity")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_executable_sequence() == other.as_executable_sequence()

    def as_executable_sequence(self):
        return (helpers.expand_exe_path("echo"), "Hello", "World!")


def create_directory(directory_path: str) -> pathlib.Path:
    """Creates the execution directory for testing."""
    tmp_dir = pathlib.Path(directory_path)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    return tmp_dir


def generate_output_files(tmp_dir: pathlib.Path):
    """Generates output and error files within the run directory for testing."""
    out_file = tmp_dir / "tmp.out"
    err_file = tmp_dir / "tmp.err"
    return out_file, err_file


def generate_directory(test_dir: str):
    """Generates a execution directory, output file, and error file for testing."""
    execution_dir = create_directory(os.path.join(test_dir, "tmp"))
    out_file, err_file = generate_output_files(execution_dir)
    return execution_dir, out_file, err_file


@pytest.fixture
def shell_launcher():
    launcher = ShellLauncher()
    yield launcher
    if any(proc.poll() is None for proc in launcher._launched.values()):
        raise RuntimeError("Test leaked processes")


@pytest.fixture
def make_shell_command(test_dir):
    run_dir, out_file_, err_file_ = generate_directory(test_dir)

    @contextlib.contextmanager
    def impl(
        args: t.Sequence[str],
        working_dir: str | os.PathLike[str] = run_dir,
        env: dict[str, str] | None = None,
        out_file: str | os.PathLike[str] = out_file_,
        err_file: str | os.PathLike[str] = err_file_,
    ):
        with (
            open(out_file, "w", encoding="utf-8") as out,
            open(err_file, "w", encoding="utf-8") as err,
        ):
            yield ShellLauncherCommand(
                env or {}, pathlib.Path(working_dir), out, err, tuple(args)
            )

    yield impl


@pytest.fixture
def shell_cmd(make_shell_command) -> ShellLauncherCommand:
    """Fixture to create an instance of Generator."""
    with make_shell_command(EchoHelloWorldEntity().as_executable_sequence()) as hello:
        yield hello


# UNIT TESTS


def test_shell_launcher_command_init(shell_cmd: ShellLauncherCommand, test_dir: str):
    """Test that ShellLauncherCommand initializes correctly"""
    assert shell_cmd.env == {}
    assert shell_cmd.path == pathlib.Path(test_dir) / "tmp"
    assert shell_cmd.stdout.name == os.path.join(test_dir, "tmp", "tmp.out")
    assert shell_cmd.stderr.name == os.path.join(test_dir, "tmp", "tmp.err")
    assert shell_cmd.command_tuple == EchoHelloWorldEntity().as_executable_sequence()


def test_shell_launcher_init(shell_launcher: ShellLauncher):
    """Test that ShellLauncher initializes correctly"""
    assert shell_launcher._launched == {}


def test_check_popen_inputs(shell_launcher: ShellLauncher, test_dir: str):
    """Test that ShellLauncher.check_popen_inputs throws correctly"""
    cmd = ShellLauncherCommand(
        {},
        pathlib.Path(test_dir) / "directory_dne",
        subprocess.DEVNULL,
        subprocess.DEVNULL,
        EchoHelloWorldEntity().as_executable_sequence(),
    )
    with pytest.raises(ValueError):
        _ = shell_launcher.start(cmd)


def test_shell_launcher_start_calls_popen(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand
):
    """Test that the process leading up to the shell launcher popen call was correct"""
    with unittest.mock.patch(
        "smartsim._core.shell.shell_launcher.sp.Popen"
    ) as mock_open:
        _ = shell_launcher.start(shell_cmd)
        mock_open.assert_called_once()


def test_shell_launcher_start_calls_popen_with_value(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand
):
    """Test that popen was called with correct values"""
    with unittest.mock.patch(
        "smartsim._core.shell.shell_launcher.sp.Popen"
    ) as mock_open:
        _ = shell_launcher.start(shell_cmd)
        mock_open.assert_called_once_with(
            shell_cmd.command_tuple,
            cwd=shell_cmd.path,
            env=shell_cmd.env,
            stdout=shell_cmd.stdout,
            stderr=shell_cmd.stderr,
        )


def test_popen_returns_popen_object(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand, test_dir: str
):
    """Test that the popen call returns a popen object"""
    id = shell_launcher.start(shell_cmd)
    with shell_launcher._launched[id] as proc:
        assert isinstance(proc, sp.Popen)


def test_popen_writes_to_output_file(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand, test_dir: str
):
    """Test that popen writes to .out file upon successful process call"""
    _, out_file, err_file = generate_directory(test_dir)
    id = shell_launcher.start(shell_cmd)
    proc = shell_launcher._launched[id]
    assert proc.wait() == 0
    assert proc.returncode == 0
    with open(out_file, "r", encoding="utf-8") as out:
        assert out.read() == "Hello World!\n"
    with open(err_file, "r", encoding="utf-8") as err:
        assert err.read() == ""


def test_popen_fails_with_invalid_cmd(shell_launcher: ShellLauncher, test_dir: str):
    """Test that popen returns a non zero returncode after failure"""
    run_dir, out_file, err_file = generate_directory(test_dir)
    with (
        open(out_file, "w", encoding="utf-8") as out,
        open(err_file, "w", encoding="utf-8") as err,
    ):
        args = (helpers.expand_exe_path("ls"), "--flag_dne")
        cmd = ShellLauncherCommand({}, run_dir, out, err, args)
        id = shell_launcher.start(cmd)
        proc = shell_launcher._launched[id]
        proc.wait()
        assert proc.returncode != 0
        with open(out_file, "r", encoding="utf-8") as out:
            assert out.read() == ""
        with open(err_file, "r", encoding="utf-8") as err:
            content = err.read()
            assert "unrecognized option" in content


def test_popen_issues_unique_ids(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand, test_dir: str
):
    """Validate that all ids are unique within ShellLauncher._launched"""
    seen = set()
    for _ in range(5):
        id = shell_launcher.start(shell_cmd)
        assert id not in seen, "Duplicate ID issued"
        seen.add(id)
    assert len(shell_launcher._launched) == 5
    assert all(proc.wait() == 0 for proc in shell_launcher._launched.values())


def test_retrieve_status_dne(shell_launcher: ShellLauncher):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    with pytest.raises(LauncherJobNotFound):
        _ = shell_launcher.get_status("dne")


def test_shell_launcher_returns_complete_status(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand
):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    for _ in range(5):
        id = shell_launcher.start(shell_cmd)
        proc = shell_launcher._launched[id]
        proc.wait()
        code = shell_launcher.get_status(id)[id]
        assert code == JobStatus.COMPLETED


def test_shell_launcher_returns_failed_status(
    shell_launcher: ShellLauncher, test_dir: str
):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    run_dir, out_file, err_file = generate_directory(test_dir)
    with (
        open(out_file, "w", encoding="utf-8") as out,
        open(err_file, "w", encoding="utf-8") as err,
    ):
        args = (helpers.expand_exe_path("ls"), "--flag_dne")
        cmd = ShellLauncherCommand({}, run_dir, out, err, args)
        for _ in range(5):
            id = shell_launcher.start(cmd)
            proc = shell_launcher._launched[id]
            proc.wait()
            code = shell_launcher.get_status(id)[id]
            assert code == JobStatus.FAILED


def test_shell_launcher_returns_running_status(
    shell_launcher: ShellLauncher, test_dir: str
):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    run_dir, out_file, err_file = generate_directory(test_dir)
    with (
        open(out_file, "w", encoding="utf-8") as out,
        open(err_file, "w", encoding="utf-8") as err,
    ):
        cmd = ShellLauncherCommand(
            {}, run_dir, out, err, (helpers.expand_exe_path("sleep"), "5")
        )
        for _ in range(5):
            id = shell_launcher.start(cmd)
            code = shell_launcher.get_status(id)[id]
            assert code == JobStatus.RUNNING
        assert all(proc.wait() == 0 for proc in shell_launcher._launched.values())


@pytest.mark.parametrize(
    "psutil_status,job_status",
    [
        pytest.param(psutil.STATUS_RUNNING, JobStatus.RUNNING, id="running"),
        pytest.param(psutil.STATUS_SLEEPING, JobStatus.RUNNING, id="sleeping"),
        pytest.param(psutil.STATUS_WAKING, JobStatus.RUNNING, id="waking"),
        pytest.param(psutil.STATUS_DISK_SLEEP, JobStatus.RUNNING, id="disk_sleep"),
        pytest.param(psutil.STATUS_DEAD, JobStatus.FAILED, id="dead"),
        pytest.param(psutil.STATUS_TRACING_STOP, JobStatus.PAUSED, id="tracing_stop"),
        pytest.param(psutil.STATUS_WAITING, JobStatus.PAUSED, id="waiting"),
        pytest.param(psutil.STATUS_STOPPED, JobStatus.PAUSED, id="stopped"),
        pytest.param(psutil.STATUS_LOCKED, JobStatus.PAUSED, id="locked"),
        pytest.param(psutil.STATUS_PARKED, JobStatus.PAUSED, id="parked"),
        pytest.param(psutil.STATUS_IDLE, JobStatus.PAUSED, id="idle"),
        pytest.param(psutil.STATUS_ZOMBIE, JobStatus.COMPLETED, id="zombie"),
        pytest.param(
            "some-brand-new-unknown-status-str", JobStatus.UNKNOWN, id="unknown"
        ),
    ],
)
def test_get_status_maps_correctly(
    psutil_status, job_status, monkeypatch: pytest.MonkeyPatch, test_dir: str
):
    """Test tht ShellLauncher.get_status returns correct mapping"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with (
        open(out_file, "w", encoding="utf-8") as out,
        open(err_file, "w", encoding="utf-8") as err,
    ):
        cmd = ShellLauncherCommand(
            {}, run_dir, out, err, EchoHelloWorldEntity().as_executable_sequence()
        )
        id = shell_launcher.start(cmd)
        proc = shell_launcher._launched[id]
        monkeypatch.setattr(proc, "poll", lambda: None)
        monkeypatch.setattr(psutil.Process, "status", lambda self: psutil_status)
        value = shell_launcher.get_status(id)
        assert value.get(id) == job_status
        assert proc.wait() == 0


@pytest.mark.parametrize(
    "args",
    (
        pytest.param(("sleep", "60"), id="Sleep for a minute"),
        *(
            pytest.param(
                (
                    sys.executable,
                    "-c",
                    textwrap.dedent(f"""\
                        import signal, time
                        signal.signal(signal.{signal_name},
                                      lambda n, f: print("Ignoring"))
                        time.sleep(60)
                        """),
                ),
                id=f"Process Swallows {signal_name}",
            )
            for signal_name in ("SIGINT", "SIGTERM")
        ),
    ),
)
def test_launcher_can_stop_processes(shell_launcher, make_shell_command, args):
    with make_shell_command(args) as cmd:
        start = time.perf_counter()
        id_ = shell_launcher.start(cmd)
        time.sleep(0.1)
        assert {id_: JobStatus.RUNNING} == shell_launcher.get_status(id_)
        assert JobStatus.FAILED == shell_launcher._stop(id_, wait_time=0.25)
        end = time.perf_counter()
        assert {id_: JobStatus.FAILED} == shell_launcher.get_status(id_)
        proc = shell_launcher._launched[id_]
        assert proc.poll() is not None
        assert proc.poll() != 0
        assert 0.1 < end - start < 1


def test_launcher_can_stop_many_processes(
    make_shell_command, shell_launcher, shell_cmd
):
    with (
        make_shell_command(("sleep", "60")) as sleep_60,
        make_shell_command(("sleep", "45")) as sleep_45,
        make_shell_command(("sleep", "30")) as sleep_30,
    ):
        id_60 = shell_launcher.start(sleep_60)
        id_45 = shell_launcher.start(sleep_45)
        id_30 = shell_launcher.start(sleep_30)
        id_short = shell_launcher.start(shell_cmd)
        time.sleep(0.1)
        assert {
            id_60: JobStatus.FAILED,
            id_45: JobStatus.FAILED,
            id_30: JobStatus.FAILED,
            id_short: JobStatus.COMPLETED,
        } == shell_launcher.stop_jobs(id_30, id_45, id_60, id_short)
