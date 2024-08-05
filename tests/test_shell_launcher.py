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
import time
import weakref
from smartsim.entity import _mock, entity, Application
from smartsim import Experiment
from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.slurm import (
    SlurmLaunchArguments,
    _as_srun_command,
)
from smartsim.settings.dispatch import sp as dsp
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

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_program_arguments() == other.as_program_arguments()

    def as_program_arguments(self):
        return ("/usr/bin/echo", "Hello", "World!")
        #return ("/usr/bin/sleep", "10")

class Files():
    def __init__(self):
        self.copy = []
        self.link = []
        self.tagged = []
        

# what is the success criteria
def test_shell_as_py(capsys):
    # a unit test should init the obj bc testing that unit of code
    launcher = ShellLauncher() # should be testing the method level
    # avoid rep
    expected_output = "hello"
    launcher.start((["echo", expected_output], "/tmp")) # use time.sleep(0.1) -> we do not need sleep in other places
    captured = capsys.readouterr()
    output = captured.out
    assert expected_output in captured.out
    # do not need to build exact str, but can just have multiple assert
    # verify echo hello
    # make a separate test for stdout and stdin -> that test only verifies one component
    # tests should do as little as possible, reduce number of constraints
    

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

# test that the process leading up to the shell launcher was corrected, integration test
# my test is identifying the change in the code
def test_shell_launcher_calls_popen(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    # monkeypatch the popen
    # create a Mock popen object
    # def my_mock_popen(*args, **kwargs):
    #     print("foo")
    # no longer care about the internals, only want to know that the process up to it was currect
    mock_popen_obj = unittest.mock.MagicMock()
    with monkeypatch.context() as ctx:
        ctx.setattr(dsp, "Popen", mock_popen_obj)
    
    # mock2 = unittest.mock.MagicMock(return_value=0) # same as monkeypatch - implements getproperty or API that looks for a unknown prop on an obj
    # mock3 = unittest.mock.MagicMock()
    # # Avoid actual network request
    # mock3.Popen = mock2
    # mock3.return_value = mock2
    env_vars = {
        "LOGGING": "verbose",
    }
    slurm_settings = LaunchSettings(launcher=LauncherType.Slurm, env_vars=env_vars)
    slurm_settings.launch_args.set_nodes(1)
    job = Job(name="jobs", entity=EchoHelloWorldEntity(), launch_settings=slurm_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    # can validate id here -> could build another mock that ensures that 22 is the pid
    id = exp.start(job)
    # mock2.assert_called_once_with(
    #     ('/usr/bin/srun', '--nodes=1', '--', '/usr/bin/echo', 'Hello', 'World!'),
    #     cwd=unittest.mock.ANY,
    #     env={},
    #     stdin=None,
    #     stdout=None
    # )
    # mock_popen_obj.assert_called()
    #mock3.assert_called_with() # the process executed the correct launcher
    # write something that makes sure the job has completed b4 the test exits
    print(id)
    #time.sleep(5) # TODO remove once blocking is added
    # asyn = concurrent, not happening in another thread, not happening somewhere else
    # focus on async io in python, make sure that anything that is io bound is async
    