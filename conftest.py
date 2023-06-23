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

import json
import os
import inspect
import shutil
import pytest
import psutil
import shutil
import smartsim
from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.settings import (
    SrunSettings,
    AprunSettings,
    JsrunSettings,
    MpirunSettings,
    RunSettings,
)
from smartsim._core.config import CONFIG
from smartsim.error import SSConfigError
from subprocess import run


# Globals, yes, but its a testing file
test_path = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(test_path, "tests", "test_output")
test_launcher = CONFIG.test_launcher
test_device = CONFIG.test_device
test_num_gpus = CONFIG.test_num_gpus
test_nic = CONFIG.test_interface
test_alloc_specs_path = os.getenv("SMARTSIM_TEST_ALLOC_SPEC_SHEET_PATH", None)
test_port = CONFIG.test_port

# Fill this at runtime if needed
test_hostlist = None


def get_account():
    global test_account
    test_account = CONFIG.test_account
    return test_account


def print_test_configuration():
    global test_path
    global test_dir
    global test_launcher
    global test_account
    global test_nic
    global test_alloc_specs_path
    global test_port
    print("TEST_SMARTSIM_LOCATION:", smartsim.__path__)
    print("TEST_PATH:", test_path)
    print("TEST_LAUNCHER:", test_launcher)
    if test_account != "":
        print("TEST_ACCOUNT:", test_account)
    print("TEST_DEVICE:", test_device)
    print("TEST_NETWORK_INTERFACE (WLM only):", test_nic)
    if test_alloc_specs_path:
        print("TEST_ALLOC_SPEC_SHEET_PATH:", test_alloc_specs_path)
    print("TEST_DIR:", test_dir)
    print("Test output will be located in TEST_DIR if there is a failure")
    print("TEST_PORT", test_port)
    print("TEST_PORT + 1", test_port + 1)


def pytest_configure():
    global test_launcher
    pytest.test_launcher = test_launcher
    pytest.wlm_options = ["slurm", "pbs", "cobalt", "lsf"]
    account = get_account()
    pytest.test_account = account


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    print_test_configuration()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    if exitstatus == 0:
        shutil.rmtree(test_dir)
    else:
        # kill all spawned processes in case of error
        kill_all_test_spawned_processes()


def kill_all_test_spawned_processes():
    # in case of test failure, clean up all spawned processes
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        # could not find parent process id
        return
    try:
        for child in parent.children(recursive=True):
            child.kill()
    except Exception:
        print("Not all processes were killed after test")


def get_hostlist():
    global test_hostlist
    if not test_hostlist:
        if "COBALT_NODEFILE" in os.environ:
            try:
                with open(os.environ["COBALT_NODEFILE"], "r") as nodefile:
                    lines = nodefile.readlines()
                    test_hostlist = list(
                        dict.fromkeys([line.strip() for line in lines])
                    )
            except:
                return None
        elif "PBS_NODEFILE" in os.environ and not shutil.which("aprun"):
            try:
                with open(os.environ["PBS_NODEFILE"], "r") as nodefile:
                    lines = nodefile.readlines()
                    test_hostlist = list(
                        dict.fromkeys([line.strip() for line in lines])
                    )
            except:
                return None
        elif "SLURM_JOB_NODELIST" in os.environ:
            try:
                nodelist = os.environ["SLURM_JOB_NODELIST"]
                test_hostlist = run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                ).stdout.split()
            except:
                return None
    return test_hostlist


@pytest.fixture(scope="session")
def alloc_specs():
    global test_alloc_specs_path
    specs = {}
    if test_alloc_specs_path:
        try:
            with open(test_alloc_specs_path) as f:
                specs = json.load(f)
        except Exception:
            raise Exception(
                (
                    f"Failed to load allocation spec sheet {test_alloc_specs_path}. "
                    "This is likely not an issue with SmartSim."
                )
            ) from None
    return specs


@pytest.fixture
def wlmutils():
    return WLMUtils


class WLMUtils:
    @staticmethod
    def set_test_launcher(new_test_launcher):
        global test_launcher
        test_launcher = new_test_launcher

    @staticmethod
    def get_test_launcher():
        global test_launcher
        return test_launcher

    @staticmethod
    def get_test_port():
        global test_port
        return test_port

    @staticmethod
    def get_test_account():
        global test_account
        return test_account

    @staticmethod
    def get_test_interface():
        global test_nic
        return test_nic

    @staticmethod
    def get_test_hostlist():
        return get_hostlist()

    @staticmethod
    def get_base_run_settings(exe, args, nodes=1, ntasks=1, **kwargs):
        if test_launcher == "slurm":
            run_args = {"--nodes": nodes, "--ntasks": ntasks, "--time": "00:10:00"}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="srun", run_args=run_args)
            return settings
        if test_launcher == "pbs":
            if shutil.which("aprun"):
                run_command = "aprun"
                run_args = {"--pes": ntasks}
            else:
                run_command = "mpirun"
                host_file = os.environ["PBS_NODEFILE"]
                run_args = {"-n": ntasks, "--hostfile": host_file}
            run_args.update(kwargs)
            settings = RunSettings(
                exe, args, run_command=run_command, run_args=run_args
            )
            return settings
        if test_launcher == "cobalt":
            if shutil.which("aprun"):
                run_command = "aprun"
                run_args = {"--pes": ntasks}
            else:
                run_command = "mpirun"
                host_file = os.environ["COBALT_NODEFILE"]
                run_args = {"-n": ntasks, "--hostfile": host_file}
            run_args.update(kwargs)
            settings = RunSettings(
                exe, args, run_command=run_command, run_args=run_args
            )
            return settings
        if test_launcher == "lsf":
            run_args = {"--np": ntasks, "--nrs": nodes}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="jsrun", run_args=run_args)
            return settings
        elif test_launcher != "local":
            raise SSConfigError(
                f"Base run settings are available for Slurm, PBS, Cobalt, and LSF, but launcher was {test_launcher}"
            )
        # TODO allow user to pick aprun vs MPIrun
        return RunSettings(exe, args)

    @staticmethod
    def get_run_settings(exe, args, nodes=1, ntasks=1, **kwargs):
        if test_launcher == "slurm":
            run_args = {"nodes": nodes, "ntasks": ntasks, "time": "00:10:00"}
            run_args.update(kwargs)
            settings = SrunSettings(exe, args, run_args=run_args)
            return settings
        elif test_launcher == "pbs":
            if shutil.which("aprun"):
                run_args = {"pes": ntasks}
                run_args.update(kwargs)
                settings = AprunSettings(exe, args, run_args=run_args)
            else:
                host_file = os.environ["PBS_NODEFILE"]
                run_args = {"n": ntasks, "hostfile": host_file}
                run_args.update(kwargs)
                settings = MpirunSettings(exe, args, run_args=run_args)
            return settings
        # TODO allow user to pick aprun vs MPIrun
        elif test_launcher == "cobalt":
            if shutil.which("aprun"):
                run_args = {"pes": ntasks}
                run_args.update(kwargs)
                settings = AprunSettings(exe, args, run_args=run_args)
            else:
                host_file = os.environ["COBALT_NODEFILE"]
                run_args = {"n": ntasks, "hostfile": host_file}
                run_args.update(kwargs)
                settings = MpirunSettings(exe, args, run_args=run_args)
            return settings
        if test_launcher == "lsf":
            run_args = {
                "nrs": nodes,
                "tasks_per_rs": max(ntasks // nodes, 1),
            }
            run_args.update(kwargs)
            settings = JsrunSettings(exe, args, run_args=run_args)
            return settings
        else:
            return RunSettings(exe, args)

    @staticmethod
    def get_orchestrator(nodes=1, batch=False):
        global test_launcher
        global test_nic
        global test_port
        if test_launcher in ["pbs", "cobalt"]:
            if not shutil.which("aprun"):
                hostlist = get_hostlist()
            else:
                hostlist = None
            db = Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                interface=test_nic,
                launcher=test_launcher,
                hosts=hostlist,
            )
        elif test_launcher == "slurm":
            db = Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                interface=test_nic,
                launcher=test_launcher,
            )
        elif test_launcher == "lsf":
            db = Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                cpus_per_shard=4,
                gpus_per_shard=2 if test_device == "GPU" else 0,
                project=get_account(),
                interface=test_nic,
                launcher=test_launcher,
            )
        else:
            db = Orchestrator(port=test_port, interface="lo")
        return db


@pytest.fixture
def local_db(fileutils, request, wlmutils):
    """Yield fixture for startup and teardown of an local orchestrator"""

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(
        caller_function=exp_name, caller_fspath=request.fspath
    )
    db = Orchestrator(port=wlmutils.get_test_port(), interface="lo")
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture
def db(fileutils, wlmutils, request):
    """Yield fixture for startup and teardown of an orchestrator"""
    launcher = wlmutils.get_test_launcher()

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(
        caller_function=exp_name, caller_fspath=request.fspath
    )
    db = wlmutils.get_orchestrator()
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture
def db_cluster(fileutils, wlmutils, request):
    """
    Yield fixture for startup and teardown of a clustered orchestrator.
    This should only be used in on_wlm and full_wlm tests.
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(
        caller_function=exp_name, caller_fspath=request.fspath
    )
    db = wlmutils.get_orchestrator(nodes=3)
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture(scope="function", autouse=True)
def environment_cleanup(monkeypatch):
    monkeypatch.delenv("SSDB", raising=False)
    monkeypatch.delenv("SSKEYIN", raising=False)
    monkeypatch.delenv("SSKEYOUT", raising=False)

@pytest.fixture
def dbutils():
    return DBUtils


class DBUtils:
    @staticmethod
    def get_db_configs():
        config_settings = {
            "enable_checkpoints": 1,
            "set_max_memory": "3gb",
            "set_eviction_strategy": "allkeys-lru",
            # set low to avoid permissions issues during testing
            # TODO make a note in SmartRedis about this method possibly
            # erroring due to the max file descriptors setting being too low
            "set_max_clients": 100,
            "set_max_message_size": 2_147_483_648,
        }
        return config_settings

    @staticmethod
    def get_smartsim_error_db_configs():
        bad_configs = {
            "save": [
                "-1",  # frequency must be positive
                "2.4",  # frequency must be specified in whole seconds
            ],
            "maxmemory": [
                "29GG",  # invalid memory form
                # str(2 ** 65) + "gb",  # memory is too much
                "3.5gb",  # invalid memory form
            ],
            "maxclients": [
                "-3",  # number clients must be positive
                str(2**65),  # number of clients is too large
                "2.9",  # number of clients must be an integer
            ],
            "proto-max-bulk-len": [
                "100",  # max message size can't be smaller than 1mb
                "9.9gb",  # invalid memory form
                "101.1",  # max message size must be an integer
            ],
            "maxmemory-policy": ["invalid-policy"],  # must use a valid maxmemory policy
            "invalid-parameter": ["99"],  # invalid key - no such configuration exists
        }
        return bad_configs

    @staticmethod
    def get_type_error_db_configs():
        bad_configs = {
            "save": [2, True, ["2"]],  # frequency must be specified as a string
            "maxmemory": [99, True, ["99"]],  # memory form must be a string
            "maxclients": [3, True, ["3"]],  # number of clients must be a string
            "proto-max-bulk-len": [
                101,
                True,
                ["101"],
            ],  # max message size must be a string
            "maxmemory-policy": [
                42,
                True,
                ["42"],
            ],  # maxmemory policies must be strings
            10: ["3"],  # invalid key - the key must be a string
        }
        return bad_configs

    @staticmethod
    def get_config_edit_method(db, config_setting):
        """Get a db configuration file edit method from a str"""
        config_edit_methods = {
            "enable_checkpoints": db.enable_checkpoints,
            "set_max_memory": db.set_max_memory,
            "set_eviction_strategy": db.set_eviction_strategy,
            "set_max_clients": db.set_max_clients,
            "set_max_message_size": db.set_max_message_size,
        }
        return config_edit_methods.get(config_setting, None)


@pytest.fixture
def fileutils():
    return FileUtils


class FileUtils:
    @staticmethod
    def _test_dir_path(caller_function, caller_fspath):
        caller_file_to_dir = os.path.splitext(str(caller_fspath))[0]
        rel_path = os.path.relpath(caller_file_to_dir, os.path.dirname(test_dir))
        dir_path = os.path.join(test_dir, rel_path, caller_function)
        return dir_path

    @staticmethod
    def get_test_dir(caller_function=None, caller_fspath=None):
        """Get path to test output.

        This function should be called without arguments from within
        a test: the returned directory will be
        `test_output/<relative_path_to_test_file>/<test_filename>/<test_name>`.
        When called from other functions (e.g. from functions in this file),
        the caller function and the caller file path should be provided.
        The directory will not be created, but the parent (and all the needed
        tree) will. This is to allow tests to create the directory.

        :param caller_function: caller function name defaults to None
        :type caller_function: str, optional
        :param caller_fspath: absolute path to file containing caller, defaults to None
        :type caller_fspath: str or Path, optional
        :return: String path to test ouptut directory
        :rtype: str
        """
        if not caller_function or not caller_fspath:
            caller_frame = inspect.stack()[1]
            caller_fspath = caller_frame.filename
            caller_function = caller_frame.function

        dir_path = FileUtils._test_dir_path(caller_function, caller_fspath)
        if not os.path.exists(os.path.dirname(dir_path)):
            os.makedirs(os.path.dirname(dir_path))
        # dir_path = os.path.join(test_dir, dir_name)
        return dir_path

    @staticmethod
    def make_test_dir(caller_function=None, caller_fspath=None):
        """Create test output directory and return path to it.

        This function should be called without arguments from within
        a test: the directory will be created as
        `test_output/<relative_path_to_test_file>/<test_filename>/<test_name>`.
        When called from other functions (e.g. from functions in this file),
        the caller function and the caller file path should be provided.

        :param caller_function: caller function name defaults to None
        :type caller_function: str, optional
        :param caller_fspath: absolute path to file containing caller, defaults to None
        :type caller_fspath: str or Path, optional
        :return: String path to test ouptut directory
        :rtype: str
        """
        if not caller_function or not caller_fspath:
            caller_frame = inspect.stack()[1]
            caller_fspath = caller_frame.filename
            caller_function = caller_frame.function

        dir_path = FileUtils._test_dir_path(caller_function, caller_fspath)
        # dir_path = os.path.join(test_dir, dir_name)
        try:
            os.makedirs(dir_path)
        except Exception:
            return dir_path
        return dir_path

    @staticmethod
    def get_test_conf_path(filename):
        file_path = os.path.join(test_path, "tests", "test_configs", filename)
        return file_path

    @staticmethod
    def get_test_dir_path(dirname):
        dir_path = os.path.join(test_path, "tests", "test_configs", dirname)
        return dir_path


@pytest.fixture
def mlutils():
    return MLUtils


class MLUtils:
    @staticmethod
    def get_test_device():
        global test_device
        return test_device

    @staticmethod
    def get_test_num_gpus():
        return test_num_gpus
