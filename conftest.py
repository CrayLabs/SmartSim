import os
import shutil
import pytest
import psutil
import smartsim
from smartsim.database import (
    CobaltOrchestrator, SlurmOrchestrator,
    PBSOrchestrator, Orchestrator,
    LSFOrchestrator
)
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import (
    SrunSettings, AprunSettings,
    JsrunSettings, RunSettings
)
from smartsim._core.config import CONFIG
from smartsim.error import SSConfigError


# Globals, yes, but its a testing file
test_path = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(test_path, "tests", "test_output")
test_launcher = CONFIG.test_launcher
test_device = CONFIG.test_device
test_nic = CONFIG.test_interface

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
    print("TEST_SMARTSIM_LOCATION:", smartsim.__path__)
    print("TEST_PATH:", test_path)
    print("TEST_LAUNCHER:", test_launcher)
    if test_account != "":
        print("TEST_ACCOUNT:", test_account)
    print("TEST_DEVICE:", test_device)
    print("TEST_NETWORK_INTERFACE (WLM only):", test_nic)
    print("TEST_DIR:", test_dir)
    print("Test output will be located in TEST_DIR if there is a failure")


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
    def get_test_account():
        global test_account
        return test_account

    @staticmethod
    def get_test_interface():
        global test_nic
        return test_nic

    @staticmethod
    def get_base_run_settings(exe, args, nodes=1, ntasks=1, **kwargs):
        if test_launcher == "slurm":
            run_args = {"--nodes": nodes,
                        "--ntasks": ntasks,
                        "--time": "00:10:00"}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="srun", run_args=run_args)
            return settings
        if test_launcher == "pbs":
            run_args = {"--pes": ntasks}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="aprun", run_args=run_args)
            return settings
        if test_launcher == "cobalt":
            run_args = {"--pes": ntasks}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="aprun", run_args=run_args)
            return settings
        if test_launcher == "lsf":
            run_args = {"--np": ntasks, "--nrs": nodes}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="jsrun", run_args=run_args)
            return settings
        elif test_launcher != "local":
            raise SSConfigError(f"Base run settings are available for Slurm, PBS, Cobalt, and LSF, but launcher was {test_launcher}")
        # TODO allow user to pick aprun vs MPIrun
        return RunSettings(exe, args)


    @staticmethod
    def get_run_settings(exe, args, nodes=1, ntasks=1, **kwargs):
        if test_launcher == "slurm":
            run_args = {"nodes": nodes,
                       "ntasks": ntasks,
                       "time": "00:10:00"}
            run_args.update(kwargs)
            settings = SrunSettings(exe, args, run_args=run_args)
            return settings
        elif test_launcher == "pbs":
            run_args = {"pes": ntasks}
            run_args.update(kwargs)
            settings = AprunSettings(exe, args, run_args=run_args)
            return settings
        # TODO allow user to pick aprun vs MPIrun
        elif test_launcher == "cobalt":
            run_args = {"pes": ntasks}
            run_args.update(kwargs)
            settings = AprunSettings(exe, args, run_args=run_args)
            return settings
        if test_launcher == "lsf":
            run_args = {"nrs": nodes,
                       "tasks_per_rs": max(ntasks//nodes,1),
                       }
            run_args.update(kwargs)
            settings = JsrunSettings(exe, args, run_args=run_args)
            return settings
        else:
            return RunSettings(exe, args)

    @staticmethod
    def get_orchestrator(nodes=1, port=6780, batch=False):
        global test_launcher
        global test_nic
        if test_launcher == "slurm":
            db = SlurmOrchestrator(db_nodes=nodes, port=port, batch=batch, interface=test_nic)
        elif test_launcher == "pbs":
            db = PBSOrchestrator(db_nodes=nodes, port=port, batch=batch, interface=test_nic)
        elif test_launcher == "cobalt":
            db = CobaltOrchestrator(db_nodes=nodes, port=port, batch=batch, interface=test_nic)
        elif test_launcher == "lsf":
            db = LSFOrchestrator(db_nodes=nodes, port=port, batch=batch, gpus_per_shard=1 if test_device=="GPU" else 0, project=get_account(), interface=test_nic)
        else:
            db = Orchestrator(port=port, interface="lo")
        return db


@pytest.fixture
def fileutils():
    return FileUtils

class FileUtils:

    @staticmethod
    def get_test_dir(dir_name):
        dir_path = os.path.join(test_dir, dir_name)
        return dir_path

    @staticmethod
    def make_test_dir(dir_name):
        dir_path = os.path.join(test_dir, dir_name)
        try:
            os.mkdir(dir_path)
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
