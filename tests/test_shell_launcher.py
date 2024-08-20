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

import tempfile
import unittest.mock
import pytest
import subprocess
import pathlib
import psutil
import difflib
import os
import uuid
import weakref
from smartsim.entity import _mock, entity, Application
from smartsim import Experiment
from smartsim.settings import LaunchSettings
from smartsim.settings.dispatch import ShellLauncher
from smartsim.settings.arguments.launch.slurm import (
    SlurmLaunchArguments,
    _as_srun_command,
)
from smartsim.status import JobStatus
from smartsim._core.utils.shell import *
from smartsim._core.commands import Command
from smartsim._core.utils import helpers
from smartsim.settings.dispatch import sp, ShellLauncher, ShellLauncherCommand
from smartsim.settings.launchCommand import LauncherType
from smartsim.launchable import Job
from smartsim.types import LaunchedJobID
from smartsim.error.errors import LauncherJobNotFound

# TODO tests bad vars in Popen call at beginning
    # tests -> helper.exe : pass in None, empty str, path with a space at beginning, a non valid command
    #       -> write a test for the invalid num of items - test_shell_launcher_fails_on_any_invalid_len_input
    #       -> have border tests for 0,1,4,6 cmd vals -> work correctly without them -> raise ValueError
        # do all of the failures as well as the sucess criteria

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
def shell_cmd(test_dir: str) -> ShellLauncherCommand:
    """Fixture to create an instance of Generator."""
    run_dir, out_file, err_file = generate_directory(test_dir)
    return ShellLauncherCommand({}, run_dir, out_file, err_file, EchoHelloWorldEntity().as_program_arguments())

# UNIT TESTS

def test_shell_launcher_command_init(shell_cmd: ShellLauncherCommand, test_dir: str):
    """Test that ShellLauncherCommand initializes correctly"""
    assert shell_cmd.env == {}
    assert shell_cmd.path == pathlib.Path(test_dir) / "tmp"
    assert shell_cmd.stdout == shell_cmd.path / "tmp.out"
    assert shell_cmd.stderr == shell_cmd.path / "tmp.err"
    assert shell_cmd.command_tuple == EchoHelloWorldEntity().as_program_arguments()

def test_shell_launcher_init():
    """Test that ShellLauncher initializes correctly"""
    shell_launcher = ShellLauncher()
    assert shell_launcher._launched == {}

def test_shell_launcher_start_calls_popen(shell_cmd: ShellLauncherCommand):
    """Test that the process leading up to the shell launcher popen call was correct"""
    shell_launcher = ShellLauncher()
    with unittest.mock.patch("smartsim.settings.dispatch.sp.Popen") as mock_open:
        _ = shell_launcher.start(shell_cmd)
        mock_open.assert_called_once()

def test_shell_launcher_start_calls_popen_with_value(shell_cmd: ShellLauncherCommand):
    """Test that popen was called with correct values"""
    shell_launcher = ShellLauncher()
    with unittest.mock.patch("smartsim.settings.dispatch.sp.Popen") as mock_open:
        _ = shell_launcher.start(shell_cmd)
        mock_open.assert_called_once_with(
            shell_cmd.command_tuple,
            cwd=shell_cmd.path,
            env=shell_cmd.env,
            stdout=shell_cmd.stdout,
            stderr=shell_cmd.stderr,
        )

def test_popen_returns_popen_object(test_dir: str):
    """Test that the popen call returns a popen object"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        cmd = ShellLauncherCommand({}, run_dir, subprocess.DEVNULL, subprocess.DEVNULL, EchoHelloWorldEntity().as_program_arguments())
        id = shell_launcher.start(cmd)
    proc = shell_launcher._launched[id]
    assert isinstance(proc, sp.Popen)


def test_popen_writes_to_output_file(test_dir: str):
    """Test that popen writes to .out file upon successful process call"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        cmd = ShellLauncherCommand({}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments())
        id = shell_launcher.start(cmd)
        val = shell_launcher.get_status(id)
        print(val)
    proc = shell_launcher._launched[id]
    # Wait for subprocess to finish
    assert proc.wait() == 0
    assert proc.returncode == 0
    with open(out_file, "r", encoding="utf-8") as out:
        assert out.read() == "Hello World!\n"
    with open(err_file, "r", encoding="utf-8") as err:
        assert err.read() == ""
    val = shell_launcher.get_status(id)
    print(val)


def test_popen_fails_with_invalid_cmd(test_dir):
    """Test that popen returns a non zero returncode after failure"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        args = (helpers.expand_exe_path("srun"), "--flag_dne")
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


def test_popen_issues_unique_ids(test_dir):
    """Validate that all ids are unique within ShellLauncher._launched"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        cmd = ShellLauncherCommand({}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments())
        for _ in range(5):
            _ = shell_launcher.start(cmd)
        assert len(shell_launcher._launched) == 5


def test_retrieve_status_dne():
    """Test tht ShellLauncher returns the status of completed Jobs"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    with pytest.raises(LauncherJobNotFound):
        _ = shell_launcher.get_status("dne")


def test_shell_launcher_returns_complete_status(test_dir):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        cmd = ShellLauncherCommand({}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments())
        for _ in range(5):
            id = shell_launcher.start(cmd)
            proc = shell_launcher._launched[id]
            proc.wait()
            code = shell_launcher.get_status(id)
            val = list(code.keys())[0]
            assert code[val] == JobStatus.COMPLETED

def test_shell_launcher_returns_failed_status(test_dir):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    # Generate testing directory
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        # Construct a invalid command to execute
        args = (helpers.expand_exe_path("srun"), "--flag_dne")
        cmd = ShellLauncherCommand({}, run_dir, out, err, args)
        # Start the execution of the command using a ShellLauncher
        for _ in range(5):
            id = shell_launcher.start(cmd)
            # Retrieve popen object
            proc = shell_launcher._launched[id]
            # Wait for subprocess to complete
            proc.wait()
            # Retrieve status of subprocess
            code = shell_launcher.get_status(id)
            val = list(code.keys())[0]
            # Assert that subprocess has completed
            assert code[val] == JobStatus.FAILED


def test_shell_launcher_returns_running_status(test_dir):
    """Test tht ShellLauncher returns the status of completed Jobs"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    # Generate testing directory
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        # Construct a command to execute
        cmd = ShellLauncherCommand({}, run_dir, out, err, (helpers.expand_exe_path("sleep"), "5"))
        # Start the execution of the command using a ShellLauncher
        for _ in range(5):
            id = shell_launcher.start(cmd)
            # Retrieve status of subprocess
            code = shell_launcher.get_status(id)
            val = list(code.keys())[0]
            # Assert that subprocess has completed
            assert code[val] == JobStatus.RUNNING
            

@pytest.mark.parametrize(
    "psutil_status,job_status",
    [
        pytest.param(psutil.STATUS_RUNNING, JobStatus.RUNNING, id="merp"),
        pytest.param(psutil.STATUS_SLEEPING, JobStatus.RUNNING, id="merp"),
        pytest.param(psutil.STATUS_WAKING, JobStatus.RUNNING, id="merp"),
        pytest.param(psutil.STATUS_DISK_SLEEP, JobStatus.RUNNING, id="merp"),
        pytest.param(psutil.STATUS_DEAD, JobStatus.FAILED, id="merp"),
        pytest.param(psutil.STATUS_TRACING_STOP, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_WAITING, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_STOPPED, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_LOCKED, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_PARKED, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_IDLE, JobStatus.PAUSED, id="merp"),
        pytest.param(psutil.STATUS_ZOMBIE, JobStatus.COMPLETED, id="merp"),
    ],
)
def test_this(psutil_status,job_status,monkeypatch: pytest.MonkeyPatch, test_dir):
    shell_launcher = ShellLauncher()
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        cmd = ShellLauncherCommand({}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments())
        id = shell_launcher.start(cmd)
        proc = shell_launcher._launched[id]
        monkeypatch.setattr(proc, "poll", lambda: None)
        monkeypatch.setattr(psutil.Process, "status", lambda self: psutil_status)
        value = shell_launcher.get_status(id)
        assert value.get(id) == job_status