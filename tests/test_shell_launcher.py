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

import os
import pathlib
import subprocess
import unittest.mock

import psutil
import pytest

from smartsim._core.shell.shellLauncher import ShellLauncher, ShellLauncherCommand, sp
from smartsim._core.utils import helpers
from smartsim._core.utils.shell import *
from smartsim.entity import _mock, entity
from smartsim.error.errors import LauncherJobNotFound
from smartsim.status import JobStatus

pytestmark = pytest.mark.group_a


class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        super().__init__("test-entity", _mock.Mock())

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_program_arguments() == other.as_program_arguments()

    def as_program_arguments(self):
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
        raise ("Test leaked processes")


@pytest.fixture
def shell_cmd(test_dir: str) -> ShellLauncherCommand:
    """Fixture to create an instance of Generator."""
    run_dir, out_file, err_file = generate_directory(test_dir)
    with (
        open(out_file, "w", encoding="utf-8") as out,
        open(err_file, "w", encoding="utf-8") as err,
    ):
        yield ShellLauncherCommand(
            {}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments()
        )


# UNIT TESTS


def test_shell_launcher_command_init(shell_cmd: ShellLauncherCommand, test_dir: str):
    """Test that ShellLauncherCommand initializes correctly"""
    assert shell_cmd.env == {}
    assert shell_cmd.path == pathlib.Path(test_dir) / "tmp"
    assert shell_cmd.stdout.name == os.path.join(test_dir, "tmp", "tmp.out")
    assert shell_cmd.stderr.name == os.path.join(test_dir, "tmp", "tmp.err")
    assert shell_cmd.command_tuple == EchoHelloWorldEntity().as_program_arguments()


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
        EchoHelloWorldEntity().as_program_arguments(),
    )
    with pytest.raises(ValueError):
        _ = shell_launcher.start(cmd)


def test_shell_launcher_start_calls_popen(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand
):
    """Test that the process leading up to the shell launcher popen call was correct"""
    with unittest.mock.patch(
        "smartsim._core.shell.shellLauncher.sp.Popen"
    ) as mock_open:
        _ = shell_launcher.start(shell_cmd)
        mock_open.assert_called_once()


def test_shell_launcher_start_calls_popen_with_value(
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand
):
    """Test that popen was called with correct values"""
    with unittest.mock.patch(
        "smartsim._core.shell.shellLauncher.sp.Popen"
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
    shell_launcher: ShellLauncher, shell_cmd: ShellLauncherCommand, test_dir: str
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
            {}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments()
        )
        id = shell_launcher.start(cmd)
        proc = shell_launcher._launched[id]
        monkeypatch.setattr(proc, "poll", lambda: None)
        monkeypatch.setattr(psutil.Process, "status", lambda self: psutil_status)
        value = shell_launcher.get_status(id)
        assert value.get(id) == job_status
        assert proc.wait() == 0
