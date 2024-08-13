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
import pathlib
import os
import uuid
import weakref
from smartsim.entity import _mock, entity, Application
from smartsim import Experiment
from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.slurm import (
    SlurmLaunchArguments,
    _as_srun_command,
)
from smartsim._core.commands import Command
from smartsim._core.utils import helpers
from smartsim.settings.dispatch import sp
from smartsim.settings.dispatch import ShellLauncher
from smartsim.settings.launchCommand import LauncherType
from smartsim.launchable import Job
from smartsim.types import LaunchedJobID
# always start with unit tests, first test shell launcher init
# make sure to test passing invalid values to shell launcher, and correct values
# verify the simple assumptions
# give each test a doc string, add comments to isolate inline

# how can I write a good test suite without being brittle - separating unit tests, group tests
# unit tests first, integration tests next, do not rely on external behav in tests
class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        path = tempfile.TemporaryDirectory()
        self._finalizer = weakref.finalize(self, path.cleanup)
        super().__init__("test-entity", _mock.Mock())
        self.files = Files()
        self.params = {}

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_program_arguments() == other.as_program_arguments()

    def as_program_arguments(self):
        return (helpers.expand_exe_path("echo"), "Hello", "World!")

class Files():
    """Represents a collection of files with different attrs for Mock entity"""
    def __init__(self):
        self.copy = []
        self.link = []
        self.tagged = []


@pytest.fixture
def experiment(monkeypatch, test_dir):
    """Fixture for creating an Experiment instance with a unique name and run directory
    for testing.
    """
    exp = Experiment(f"test-exp-{uuid.uuid4()}", test_dir)
    # Generate run directory 
    run_dir = pathlib.Path(test_dir) / "tmp"
    run_dir.mkdir(exist_ok=True, parents=True)
    # Generate out / err files 
    out_file = run_dir / "tmp.out"
    err_file = run_dir / "tmp.err"
    out_file.touch()
    err_file.touch()
    # MonkeyPatch Experiment._generate
    monkeypatch.setattr(exp, "_generate", lambda gen, job, idx: (run_dir, out_file, err_file))
    yield exp

# popen returns a non 0 when an error occurs, so test invalid path
# assert not id, might retry -> assert called 5 times, -> could verify that a warning was printed

# should test a success cond, a failure condition


# UNIT TESTS

def test_shell_launcher_init():
    """A simple test to validate that ShellLauncher correctly initializes"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    # Assert that private attribute is expected value
    assert shell_launcher._launched == {}


def test_shell_launcher_calls_popen():
    """Test that the process leading up to the shell launcher popen call was corrected"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    # Mock command passed to ShellLauncher.start
    cmd = Command(["env_vars", "run_dir", "out_file_path", "err_file_path", EchoHelloWorldEntity().as_program_arguments()])
    # Setup mock for Popen class from the smartsim.settings.dispatch.sp module
    # to temporarily replace the actual Popen class with a mock version
    with unittest.mock.patch("smartsim.settings.dispatch.sp.Popen") as mock_open:
        # Assign a mock value of 12345 to the process id attr of the mocked Popen object
        mock_open.pid = unittest.mock.MagicMock(return_value=12345)
        # Assign a mock return value of 0 to the returncode attr of the mocked Popen object
        mock_open.returncode = unittest.mock.MagicMock(return_value=0)
        # Execute Experiment.start
        _ = shell_launcher.start(cmd)
        # Assert that the mock_open object was called during the execution of the Experiment.start
        mock_open.assert_called_once()

def test_this(test_dir: str):
    """Test that popen was called with correct types"""
    job = Job(name="jobs", entity=EchoHelloWorldEntity(), launch_settings=LaunchSettings(launcher=LauncherType.Slurm))
    exp = Experiment(name="exp_name", exp_path=test_dir)
    # Setup mock for Popen class from the smartsim.settings.dispatch.sp module
    # to temporarily replace the actual Popen class with a mock version
    with unittest.mock.patch("smartsim.settings.dispatch.sp.Popen") as mock_open:
        # Assign a mock value of 12345 to the pid attr of the mocked Popen object
        mock_open.pid = unittest.mock.MagicMock(return_value=12345)
        # Assign a mock return value of 0 to the returncode attr of the mocked Popen object
        mock_open.returncode = unittest.mock.MagicMock(return_value=0)
        _ = exp.start(job)
        # Assert that the mock_open object was called during the execution of the Experiment.start with value
        mock_open.assert_called_once_with(
            (helpers.expand_exe_path("srun"), '--', helpers.expand_exe_path("echo"), 'Hello', 'World!'),
            cwd=unittest.mock.ANY,
            env={},
            stdin=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
        )

def create_directory(directory_path) -> pathlib.Path:
    """Creates the execution directory for testing."""
    tmp_dir = pathlib.Path(directory_path)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    return tmp_dir

def generate_output_files(tmp_dir):
    """Generates output and error files within the run directory for testing."""
    out_file = tmp_dir / "tmp.out"
    err_file = tmp_dir / "tmp.err"
    return out_file, err_file

def generate_directory(test_dir):
    """Generates a execution directory, output file, and error file for testing."""
    execution_dir = create_directory(os.path.join(test_dir, "/tmp"))
    out_file, err_file = generate_output_files(execution_dir)
    return execution_dir, out_file, err_file

def test_popen_writes_to_out(test_dir):
    """TODO"""
    # Init ShellLauncher
    shell_launcher = ShellLauncher()
    # Generate testing directory
    run_dir, out_file, err_file = generate_directory(test_dir)
    with open(out_file, "w", encoding="utf-8") as out, open(err_file, "w", encoding="utf-8") as err:
        # Construct a command to execute
        cmd = Command([{}, run_dir, out, err, EchoHelloWorldEntity().as_program_arguments()])
        # Start the execution of the command using a ShellLauncher
        id = shell_launcher.start(cmd)
    # Retrieve the process associated with the launched command
    proc = shell_launcher._launched[id]
    # Check successful execution
    assert proc.wait() == 0
    # Reopen out_file in read mode
    with open(out_file, "r", encoding="utf-8") as out:
        # Assert that the content of the output file is expected
        assert out.read() == "Hello World!\n"

    












# write something that makes sure the job has completed b4 the test exits
# print(id)
#time.sleep(5) # TODO remove once blocking is added
# asyn = concurrent, not happening in another thread, not happening somewhere else
# focus on async io in python, make sure that anything that is io bound is async

# what is the success criteria
# def test_shell_as_py(capsys):
#     # a unit test should init the obj bc testing that unit of code
#     launcher = ShellLauncher() # should be testing the method level
#     # avoid rep
#     expected_output = "hello"
#     launcher.start((["echo", expected_output], "/tmp")) # use time.sleep(0.1) -> we do not need sleep in other places
#     captured = capsys.readouterr()
#     output = captured.out
#     assert expected_output in captured.out
    # do not need to build exact str, but can just have multiple assert
    # verify echo hello
    # make a separate test for stdout and stdin -> that test only verifies one component
    # tests should do as little as possible, reduce number of constraints