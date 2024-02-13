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

import json
import os
import pytest
import psutil
import shutil
import smartsim
from smartsim import Experiment
from smartsim.entity import Model
from smartsim.database import Orchestrator
from smartsim.settings import (
    SrunSettings,
    AprunSettings,
    JsrunSettings,
    MpirunSettings,
    MpiexecSettings,
    PalsMpiexecSettings,
    RunSettings,
)
from smartsim._core.config import CONFIG
from smartsim.error import SSConfigError
from subprocess import run
import sys
import tempfile
import typing as t
import uuid
import warnings


# pylint: disable=redefined-outer-name,invalid-name,global-statement

# Globals, yes, but its a testing file
test_path = os.path.dirname(os.path.abspath(__file__))
test_output_root = os.path.join(test_path, "tests", "test_output")
test_launcher = CONFIG.test_launcher
test_device = CONFIG.test_device.upper()
test_num_gpus = CONFIG.test_num_gpus
test_nic = CONFIG.test_interface
test_alloc_specs_path = os.getenv("SMARTSIM_TEST_ALLOC_SPEC_SHEET_PATH", None)
test_port = CONFIG.test_port
test_account = CONFIG.test_account or ""
test_batch_resources: t.Dict[t.Any,t.Any] = CONFIG.test_batch_resources

# Fill this at runtime if needed
test_hostlist = None
has_aprun = shutil.which("aprun") is not None

def get_account() -> str:
    return test_account


def print_test_configuration() -> None:
    print("TEST_SMARTSIM_LOCATION:", smartsim.__path__)
    print("TEST_PATH:", test_path)
    print("TEST_LAUNCHER:", test_launcher)
    if test_account != "":
        print("TEST_ACCOUNT:", test_account)
    test_device_msg = f"TEST_DEVICE: {test_device}"
    if test_device == "GPU":
        test_device_msg += f"x{test_num_gpus}"
    print(test_device_msg)
    print("TEST_NETWORK_INTERFACE (WLM only):", test_nic)
    if test_alloc_specs_path:
        print("TEST_ALLOC_SPEC_SHEET_PATH:", test_alloc_specs_path)
    print("TEST_DIR:", test_output_root)
    print("Test output will be located in TEST_DIR if there is a failure")
    print(
        "TEST_PORTS:", ", ".join(str(port) for port in range(test_port, test_port + 3))
    )
    if test_batch_resources:
        print("TEST_BATCH_RESOURCES: ")
        print(json.dumps(test_batch_resources, indent=2))


def pytest_configure() -> None:
    pytest.test_launcher = test_launcher
    pytest.wlm_options = ["slurm", "pbs", "lsf", "pals"]
    account = get_account()
    pytest.test_account = account
    pytest.test_device = test_device
    pytest.has_aprun = has_aprun


def pytest_sessionstart(
    session: pytest.Session,  # pylint: disable=unused-argument
) -> None:
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    if os.path.isdir(test_output_root):
        shutil.rmtree(test_output_root)
    os.makedirs(test_output_root)
    print_test_configuration()


def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int  # pylint: disable=unused-argument
) -> None:
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    if exitstatus == 0:
        shutil.rmtree(test_output_root)
    else:
        # kill all spawned processes in case of error
        kill_all_test_spawned_processes()


def kill_all_test_spawned_processes() -> None:
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


def get_hostlist() -> t.Optional[t.List[str]]:
    global test_hostlist
    if not test_hostlist:
        if "PBS_NODEFILE" in os.environ and test_launcher == "pals":
            # with PALS, we need a hostfile even if `aprun` is available
            try:
                return _parse_hostlist_file(os.environ["PBS_NODEFILE"])
            except FileNotFoundError:
                return None
        elif "PBS_NODEFILE" in os.environ and not shutil.which("aprun"):
            try:
                return _parse_hostlist_file(os.environ["PBS_NODEFILE"])
            except FileNotFoundError:
                return None
        elif "SLURM_JOB_NODELIST" in os.environ:
            try:
                nodelist = os.environ["SLURM_JOB_NODELIST"]
                test_hostlist = run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.split()
            except Exception:
                return None
    return test_hostlist


def _parse_hostlist_file(path: str) -> t.List[str]:
    with open(path, "r", encoding="utf-8") as nodefile:
        return list({line.strip() for line in nodefile.readlines()})


@pytest.fixture(scope="session")
def alloc_specs() -> t.Dict[str, t.Any]:
    specs: t.Dict[str, t.Any] = {}
    if test_alloc_specs_path:
        try:
            with open(test_alloc_specs_path, encoding="utf-8") as spec_file:
                specs = json.load(spec_file)
        except Exception:
            raise Exception(
                (
                    f"Failed to load allocation spec sheet {test_alloc_specs_path}. "
                    "This is likely not an issue with SmartSim."
                )
            ) from None
    return specs


@pytest.fixture
def wlmutils() -> t.Type[WLMUtils]:
    return WLMUtils


class WLMUtils:
    @staticmethod
    def set_test_launcher(new_test_launcher: str) -> None:
        global test_launcher
        test_launcher = new_test_launcher

    @staticmethod
    def get_test_launcher() -> str:
        return test_launcher

    @staticmethod
    def get_test_port() -> int:
        return test_port

    @staticmethod
    def get_test_account() -> str:
        return get_account()

    @staticmethod
    def get_test_interface() -> t.List[str]:
        return test_nic

    @staticmethod
    def get_test_hostlist() -> t.Optional[t.List[str]]:
        return get_hostlist()

    @staticmethod
    def get_batch_resources() -> t.Dict:
        return test_batch_resources

    @staticmethod
    def get_base_run_settings(
        exe: str, args: t.List[str], nodes: int = 1, ntasks: int = 1, **kwargs: t.Any
    ) -> RunSettings:
        run_args: t.Dict[str, t.Union[int, str, float, None]] = {}

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
        if test_launcher == "pals":
            host_file = os.environ["PBS_NODEFILE"]
            run_args = {"--np": ntasks, "--hostfile": host_file}
            run_args.update(kwargs)
            return RunSettings(exe, args, run_command="mpiexec", run_args=run_args)
        if test_launcher == "lsf":
            run_args = {"--np": ntasks, "--nrs": nodes}
            run_args.update(kwargs)
            settings = RunSettings(exe, args, run_command="jsrun", run_args=run_args)
            return settings
        if test_launcher != "local":
            raise SSConfigError(
                "Base run settings are available for Slurm, PBS, "
                f"and LSF, but launcher was {test_launcher}"
            )
        # TODO allow user to pick aprun vs MPIrun
        return RunSettings(exe, args)

    @staticmethod
    def get_run_settings(
        exe: str, args: t.List[str], nodes: int = 1, ntasks: int = 1, **kwargs: t.Any
    ) -> RunSettings:
        run_args: t.Dict[str, t.Union[int, str, float, None]] = {}

        if test_launcher == "slurm":
            run_args = {"nodes": nodes, "ntasks": ntasks, "time": "00:10:00"}
            run_args.update(kwargs)
            return SrunSettings(exe, args, run_args=run_args)
        if test_launcher == "pbs":
            if shutil.which("aprun"):
                run_args = {"pes": ntasks}
                run_args.update(kwargs)
                return AprunSettings(exe, args, run_args=run_args)

            host_file = os.environ["PBS_NODEFILE"]
            run_args = {"n": ntasks, "hostfile": host_file}
            run_args.update(kwargs)
            return MpirunSettings(exe, args, run_args=run_args)
        if test_launcher == "pals":
            host_file = os.environ["PBS_NODEFILE"]
            run_args = {"np": ntasks, "hostfile": host_file}
            run_args.update(kwargs)
            return PalsMpiexecSettings(exe, args, run_args=run_args)
        if test_launcher == "lsf":
            run_args = {
                "nrs": nodes,
                "tasks_per_rs": max(ntasks // nodes, 1),
            }
            run_args.update(kwargs)
            return JsrunSettings(exe, args, run_args=run_args)

        return RunSettings(exe, args)

    @staticmethod
    def get_orchestrator(nodes: int = 1, batch: bool = False) -> Orchestrator:
        if test_launcher == "pbs":
            if not shutil.which("aprun"):
                hostlist = get_hostlist()
            else:
                hostlist = None
            return Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                interface=test_nic,
                launcher=test_launcher,
                hosts=hostlist,
            )
        if test_launcher == "pals":
            hostlist = get_hostlist()
            return Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                interface=test_nic,
                launcher=test_launcher,
                hosts=hostlist,
            )
        if test_launcher == "slurm":
            return Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                interface=test_nic,
                launcher=test_launcher,
            )
        if test_launcher == "lsf":
            return Orchestrator(
                db_nodes=nodes,
                port=test_port,
                batch=batch,
                cpus_per_shard=4,
                gpus_per_shard=2 if test_device == "GPU" else 0,
                project=get_account(),
                interface=test_nic,
                launcher=test_launcher,
            )

        return Orchestrator(port=test_port, interface="lo")

    @staticmethod
    def choose_host(rs: RunSettings) -> t.Optional[str]:
        if isinstance(rs, (MpirunSettings, MpiexecSettings)):
            hl = get_hostlist()
            if hl is not None:
                return hl[0]

        return None

@pytest.fixture
def local_db(
    request: t.Any, wlmutils: t.Type[WLMUtils], test_dir: str
) -> t.Generator[Orchestrator, None, None]:
    """Yield fixture for startup and teardown of an local orchestrator"""

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    db = Orchestrator(port=wlmutils.get_test_port(), interface="lo")
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture
def db(
    request: t.Any, wlmutils: t.Type[WLMUtils], test_dir: str
) -> t.Generator[Orchestrator, None, None]:
    """Yield fixture for startup and teardown of an orchestrator"""
    launcher = wlmutils.get_test_launcher()

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    db = wlmutils.get_orchestrator()
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture
def db_cluster(
    test_dir: str, wlmutils: t.Type[WLMUtils], request: t.Any
) -> t.Generator[Orchestrator, None, None]:
    """
    Yield fixture for startup and teardown of a clustered orchestrator.
    This should only be used in on_wlm and full_wlm tests.
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = request.function.__name__
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    db = wlmutils.get_orchestrator(nodes=3)
    db.set_path(test_dir)
    exp.start(db)

    yield db
    # pass or fail, the teardown code below is ran after the
    # completion of a test case that uses this fixture
    exp.stop(db)


@pytest.fixture(scope="function", autouse=True)
def environment_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ.keys():
        if key.startswith("SSDB"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SSKEYIN", raising=False)
    monkeypatch.delenv("SSKEYOUT", raising=False)


@pytest.fixture
def dbutils() -> t.Type[DBUtils]:
    return DBUtils


class DBUtils:
    @staticmethod
    def get_db_configs() -> t.Dict[str, t.Any]:
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
    def get_smartsim_error_db_configs() -> t.Dict[str, t.Any]:
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
    def get_type_error_db_configs() -> t.Dict[t.Union[int, str], t.Any]:
        bad_configs: t.Dict[t.Union[int, str], t.Any] = {
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
    def get_config_edit_method(
        db: Orchestrator, config_setting: str
    ) -> t.Optional[t.Callable[..., None]]:
        """Get a db configuration file edit method from a str"""
        config_edit_methods: t.Dict[str, t.Callable[..., None]] = {
            "enable_checkpoints": db.enable_checkpoints,
            "set_max_memory": db.set_max_memory,
            "set_eviction_strategy": db.set_eviction_strategy,
            "set_max_clients": db.set_max_clients,
            "set_max_message_size": db.set_max_message_size,
        }
        return config_edit_methods.get(config_setting, None)


def _sanitize_caller_function(caller_function: str) -> str:
    # Parametrized test functions end with a list of all
    # parameter values. The list is enclosed in square brackets.
    # We split at the opening bracket, sanitize the string
    # to its right and then merge the function name and
    # the sanitized list with a dot.
    caller_function = caller_function.replace("]","")
    caller_function_list = caller_function.split("[", maxsplit=1)

    def is_accepted_char(char: str) -> bool:
        return char.isalnum() or char in "-._"

    if len(caller_function_list) > 1:
        caller_function_list[1] = "".join(
            filter(is_accepted_char, caller_function_list[1])
        )

    return ".".join(caller_function_list)


@pytest.fixture
def test_dir(request: pytest.FixtureRequest) -> str:
    caller_function = _sanitize_caller_function(request.node.name)
    dir_path = FileUtils.get_test_output_path(caller_function, str(request.path))

    try:
        os.makedirs(dir_path)
    except Exception:
        return dir_path
    return dir_path


@pytest.fixture
def fileutils() -> t.Type[FileUtils]:
    return FileUtils


class FileUtils:
    @staticmethod
    def get_test_output_path(caller_function: str, caller_fspath: str) -> str:
        caller_file_to_dir = os.path.splitext(str(caller_fspath))[0]
        rel_path = os.path.relpath(caller_file_to_dir, os.path.dirname(test_output_root))
        dir_path = os.path.join(test_output_root, rel_path, caller_function)
        return dir_path

    @staticmethod
    def get_test_conf_path(filename: str) -> str:
        file_path = os.path.join(test_path, "tests", "test_configs", filename)
        return file_path

    @staticmethod
    def get_test_dir_path(dirname: str) -> str:
        dir_path = os.path.join(test_path, "tests", "test_configs", dirname)
        return dir_path

    @staticmethod
    def make_test_file(file_name: str, file_dir: str, file_content: t.Optional[str] = None) -> str:
        """Create a dummy file in the test output directory.

        :param file_name: name of file to create, e.g. "file.txt"
        :type file_name: str
        :param file_dir: path
        :type file_dir: str
        :return: String path to test output file
        :rtype: str
        """
        file_path = os.path.join(file_dir, file_name)
        os.makedirs(file_dir)
        with open(file_path, "w+", encoding="utf-8") as dummy_file:
            if not file_content:
                dummy_file.write("dummy\n")
            else:
                dummy_file.write(file_content)

        return file_path


@pytest.fixture
def mlutils() -> t.Type[MLUtils]:
    return MLUtils


class MLUtils:
    @staticmethod
    def get_test_device() -> str:
        return test_device

    @staticmethod
    def get_test_num_gpus() -> int:
        return test_num_gpus


@pytest.fixture
def coloutils() -> t.Type[ColoUtils]:
    return ColoUtils


class ColoUtils:
    @staticmethod
    def setup_test_colo(
        fileutils: t.Type[FileUtils],
        db_type: str,
        exp: Experiment,
        application_file: str,
        db_args: t.Dict[str, t.Any],
        colo_settings: t.Optional[RunSettings] = None,
        colo_model_name: str = "colocated_model",
        port: int = test_port,
        on_wlm: bool = False,
    ) -> Model:
        """Setup database needed for the colo pinning tests"""

        # get test setup
        sr_test_script = fileutils.get_test_conf_path(application_file)

        # Create an app with a colo_db which uses 1 db_cpu
        if colo_settings is None:
            colo_settings = exp.create_run_settings(
                exe=sys.executable, exe_args=[sr_test_script]
            )
        if on_wlm:
            colo_settings.set_tasks(1)
            colo_settings.set_nodes(1)
        colo_model = exp.create_model(colo_model_name, colo_settings)

        if db_type in ["tcp", "deprecated"]:
            db_args["port"] = port
            db_args["ifname"] = "lo"
        if db_type == "uds" and colo_model_name is not None:
            tmp_dir = tempfile.gettempdir()
            socket_suffix = str(uuid.uuid4())[:7]
            db_args["unix_socket"] = os.path.join(tmp_dir,
                f"{colo_model_name}_{socket_suffix}.socket")

        colocate_fun: t.Dict[str, t.Callable[..., None]] = {
            "tcp": colo_model.colocate_db_tcp,
            "deprecated": colo_model.colocate_db,
            "uds": colo_model.colocate_db_uds,
        }
        with warnings.catch_warnings():
            if db_type == "deprecated":
                warnings.filterwarnings(
                    "ignore",
                    message="`colocate_db` has been deprecated"
                )
            colocate_fun[db_type](**db_args)
        # assert model will launch with colocated db
        assert colo_model.colocated
        # Check to make sure that limit_db_cpus made it into the colo settings
        return colo_model

@pytest.fixture
def config() -> smartsim._core.config.Config:
    return CONFIG
