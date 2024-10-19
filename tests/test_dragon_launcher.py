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
import multiprocessing as mp
import os
import pathlib
import sys
import time
import typing as t
from unittest.mock import MagicMock

import pytest
import zmq

import smartsim._core.config
from smartsim._core._cli.scripts.dragon_install import (
    DEFAULT_DRAGON_VERSION,
    create_dotenv,
)
from smartsim._core.config.config import get_config
from smartsim._core.launcher.dragon.dragonLauncher import (
    DragonConnector,
    DragonLauncher,
)
from smartsim._core.launcher.dragon.dragonSockets import (
    get_authenticator,
    get_secure_socket,
)
from smartsim._core.launcher.step.dragonStep import DragonBatchStep, DragonStep
from smartsim._core.schemas.dragonRequests import DragonBootstrapRequest
from smartsim._core.schemas.dragonResponses import (
    DragonHandshakeResponse,
    DragonRunResponse,
)
from smartsim._core.utils.network import IFConfig, find_free_port
from smartsim._core.utils.security import KeyManager
from smartsim.error.errors import LauncherError
from smartsim.settings.dragonRunSettings import DragonRunSettings
from smartsim.settings.slurmSettings import SbatchSettings


@pytest.fixture
def dragon_batch_step(test_dir: str) -> DragonBatchStep:
    """Fixture for creating a default batch of steps for a dragon launcher"""
    test_path = pathlib.Path(test_dir)

    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    # ensure the status_dir is set
    status_dir = (test_path / ".smartsim" / "logs").as_posix()
    batch_step.meta["status_dir"] = status_dir

    # create some steps to verify the requests file output changes
    rs0 = DragonRunSettings(exe="sleep", exe_args=["1"])
    rs1 = DragonRunSettings(exe="sleep", exe_args=["2"])
    rs2 = DragonRunSettings(exe="sleep", exe_args=["3"])
    rs3 = DragonRunSettings(exe="sleep", exe_args=["4"])

    names = "test00", "test01", "test02", "test03"
    settings = rs0, rs1, rs2, rs3

    # create steps with:
    # no affinity, cpu affinity only, gpu affinity only, cpu and gpu affinity
    cpu_affinities = [[], [0, 1, 2], [], [3, 4, 5, 6]]
    gpu_affinities = [[], [], [0, 1, 2], [3, 4, 5, 6]]

    # assign some unique affinities to each run setting instance
    for index, rs in enumerate(settings):
        if gpu_affinities[index]:
            rs.set_node_feature("gpu")
        rs.set_cpu_affinity(cpu_affinities[index])
        rs.set_gpu_affinity(gpu_affinities[index])

    steps = list(
        DragonStep(name_, test_dir, rs_) for name_, rs_ in zip(names, settings)
    )

    for index, step in enumerate(steps):
        # ensure meta is configured...
        step.meta["status_dir"] = status_dir
        # ... and put all the steps into the batch
        batch_step.add_to_batch(steps[index])

    return batch_step


# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


is_mac = sys.platform == "darwin"


class MockPopen:
    calls = []

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self.args = args
        self.kwargs = kwargs

        MockPopen.calls.append((args, kwargs))

    @property
    def pid(self) -> int:
        return 99999

    @property
    def returncode(self) -> int:
        return 0

    @property
    def stdout(self):
        return None

    @property
    def stderr(self):
        return None

    def wait(self, timeout: float) -> None:
        time.sleep(timeout)


class MockSocket:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self._bind_address = ""

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self

    def bind(self, addr: str) -> None:
        self._bind_address = addr

    def recv_string(self, flags: int) -> str:
        dbr = DragonBootstrapRequest(address=self._bind_address)
        return f"bootstrap|{dbr.json()}"

    def close(self) -> None: ...

    def send(self, *args, **kwargs) -> None: ...

    def send_json(self, json: str) -> None: ...

    def send_string(*args, **kwargs) -> None: ...

    def connect(*args, **kwargs) -> None: ...

    @property
    def bind_address(self) -> str:
        return self._bind_address


class MockAuthenticator:
    def __init__(self, context: zmq.Context, log: t.Any) -> None:
        self.num_starts: int = 0
        self.num_stops: int = 0
        self.num_configure_curves: int = 0
        self.context = context
        self.thread = None

    def configure_curve(self, *args, **kwargs) -> None:
        self.cfg_args = args
        self.cfg_kwargs = kwargs
        self.num_configure_curves += 1

    def start(self) -> None:
        self.num_starts += 1

    def stop(self) -> None:
        self.num_stops += 1

    def is_alive(self) -> bool:
        return self.num_starts > 0 and self.num_stops == 0


def mock_dragon_env(test_dir, *args, **kwargs):
    """Create a mock dragon environment that can talk to the launcher through ZMQ"""
    logger = logging.getLogger(__name__)
    config = get_config()
    logging.basicConfig(level=logging.DEBUG)
    try:
        addr = "127.0.0.1"
        callback_port = kwargs["port"]
        head_port = find_free_port(start=callback_port + 1)
        context = zmq.Context.instance()
        context.setsockopt(zmq.SNDTIMEO, config.dragon_server_timeout)
        context.setsockopt(zmq.RCVTIMEO, config.dragon_server_timeout)
        authenticator = get_authenticator(context, -1)

        callback_socket = get_secure_socket(context, zmq.REQ, False)
        dragon_head_socket = get_secure_socket(context, zmq.REP, True)

        full_addr = f"{addr}:{callback_port}"
        callback_socket.connect(f"tcp://{full_addr}")

        full_head_addr = f"tcp://{addr}:{head_port}"
        dragon_head_socket.bind(full_head_addr)

        req = DragonBootstrapRequest(address=full_head_addr)

        msg_sent = False
        while not msg_sent:
            logger.info("Sending bootstrap request to callback socket")
            callback_socket.send_string("bootstrap|" + req.json())
            # hold until bootstrap response is received
            logger.info("Receiving bootstrap response from callback socket")
            _ = callback_socket.recv()
            msg_sent = True

        hand_shaken = False
        while not hand_shaken:
            # other side should set up a socket and push me a `HandshakeRequest`
            logger.info("Receiving handshake request through dragon head socket")
            _ = dragon_head_socket.recv()
            # acknowledge handshake success w/DragonHandshakeResponse
            logger.info("Sending handshake response through dragon head socket")
            handshake_ack = DragonHandshakeResponse(dragon_pid=os.getpid())
            dragon_head_socket.send_string(f"handshake|{handshake_ack.json()}")

            hand_shaken = True

        shutting_down = False
        while not shutting_down:
            logger.info("Waiting for shutdown request through dragon head socket")
            # any incoming request at this point in test is my shutdown...
            try:
                message = dragon_head_socket.recv()
                logger.info(f"Received final message {message}")
            finally:
                shutting_down = True
        try:
            logger.info("Handshake complete. Shutting down mock dragon env.")
            authenticator.stop()
        finally:
            logger.info("Dragon mock env exiting...")

    except Exception as ex:
        logger.info(f"exception occurred while configuring mock handshaker: {ex}")
        raise ex from None


def test_dragon_connect_attributes(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Test the connection to a dragon environment dynamically selects an open port
    in the range supplied and passes the correct environment"""
    test_path = pathlib.Path(test_dir)

    with monkeypatch.context() as ctx:
        # make sure we don't touch "real keys" during a test
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        mock_socket = MockSocket()

        # look at test_dir for dragon config
        ctx.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
        # avoid finding real interface
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonConnector.get_best_interface_and_address",
            lambda: IFConfig(interface="faux_interface", address="127.0.0.1"),
        )
        # we need to set the socket value or is_connected returns False
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonLauncher.DragonConnector._handshake",
            lambda self, address: ...,
        )
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)
        # avoid starting a real zmq socket
        ctx.setattr("zmq.Context.socket", mock_socket)
        # avoid starting a real process for dragon entrypoint
        ctx.setattr(
            "subprocess.Popen", lambda *args, **kwargs: MockPopen(*args, **kwargs)
        )

        # avoid reading "real" config in test...
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)
        dotenv_path = smartsim._core.config.CONFIG.dragon_dotenv
        dotenv_path.parent.mkdir(parents=True)
        dotenv_path.write_text("FOO=BAR\nBAZ=BOO")

        dragon_connector = DragonConnector()
        dragon_connector.connect_to_dragon()

        chosen_port = int(mock_socket.bind_address.split(":")[-1])
        assert chosen_port >= 5995

        # grab the kwargs env=xxx from the mocked popen to check what was passed
        env = MockPopen.calls[0][1].get("env", None)

        # confirm the environment values were passed from .env file to dragon process
        assert "PYTHONUNBUFFERED" in env
        assert "FOO" in env
        assert "BAZ" in env

        dragon_connector._authenticator.stop()


@pytest.mark.parametrize(
    "socket_type, is_server",
    [
        pytest.param(zmq.REQ, True, id="as-server"),
        pytest.param(zmq.REP, False, id="as-client"),
    ],
)
def test_secure_socket_authenticator_setup(
    test_dir: str, monkeypatch: pytest.MonkeyPatch, socket_type: int, is_server: bool
):
    """Ensure the authenticator created by the secure socket factory method
    is fully configured and started when returned to a client"""

    with monkeypatch.context() as ctx:
        # look at test dir for dragon config
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)

        authenticator = get_authenticator(zmq.Context.instance())

        km = KeyManager(get_config(), as_server=is_server)

        assert isinstance(authenticator, MockAuthenticator)

        # ensure authenticator was configured
        assert authenticator.num_configure_curves > 0
        # ensure authenticator was started
        assert authenticator.num_starts > 0
        assert authenticator.context == zmq.Context.instance()
        # ensure authenticator will accept any secured connection
        assert authenticator.cfg_kwargs.get("domain", "") == "*"
        # ensure authenticator is using the expected set of keys
        assert authenticator.cfg_kwargs.get("location", "") == km.client_keys_dir

        authenticator.stop()


@pytest.mark.parametrize(
    "as_server",
    [
        pytest.param(True, id="server-socket"),
        pytest.param(False, id="client-socket"),
    ],
)
def test_secure_socket_setup(
    test_dir: str, monkeypatch: pytest.MonkeyPatch, as_server: bool
):
    """Ensure the authenticator created by the secure socket factory method
    is fully configured and started when returned to a client"""

    with monkeypatch.context() as ctx:
        # look at test dir for dragon config
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)

        context = zmq.Context.instance()

        socket = get_secure_socket(context, zmq.REP, as_server)

        # verify the socket is correctly configured to use curve authentication
        assert bool(socket.CURVE_SERVER) == as_server
        assert not socket.closed

        socket.close()


def test_secure_socket(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Ensure the authenticator created by the secure socket factory method
    is fully configured and started when returned to a client"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    with monkeypatch.context() as ctx:
        # make sure we don't touch "real keys" during a test
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)
        context = zmq.Context.instance()
        authenticator = get_authenticator(context)
        server = get_secure_socket(context, zmq.REP, True)

        ip, port = "127.0.0.1", find_free_port(start=9999)

        try:
            server.bind(f"tcp://*:{port}")

            client = get_secure_socket(context, zmq.REQ, False)

            client.connect(f"tcp://{ip}:{port}")

            to_send = "you get a foo! you get a foo! everybody gets a foo!"
            client.send_string(to_send, flags=zmq.NOBLOCK)

            received_msg = server.recv_string()
            assert received_msg == to_send
            logger.debug(f"server received: {received_msg}")
        finally:
            if authenticator:
                authenticator.stop()
            if client:
                client.close()
            if server:
                server.close()


@pytest.mark.skipif(is_mac, reason="unsupported on MacOSX")
def test_dragon_launcher_handshake(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Test that a real handshake between a launcher & dragon environment
    completes successfully using secure sockets"""
    addr = "127.0.0.1"
    bootstrap_port = find_free_port(start=5995)

    with monkeypatch.context() as ctx:
        # make sure we don't touch "real keys" during a test
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        # look at test dir for dragon config
        ctx.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
        # avoid finding real interface since we may not be on a super
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonConnector.get_best_interface_and_address",
            lambda: IFConfig("faux_interface", addr),
        )

        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonConnector._dragon_cleanup",
            lambda server_socket, server_process_pid, server_authenticator: server_authenticator.stop(),
        )

        # start up a faux dragon env that knows how to do the handshake process
        # but uses secure sockets for all communication.
        mock_dragon = mp.Process(
            target=mock_dragon_env,
            daemon=True,
            kwargs={"port": bootstrap_port, "test_dir": test_dir},
        )

        def fn(*args, **kwargs):
            mock_dragon.start()
            return mock_dragon

        ctx.setattr("subprocess.Popen", fn)

        connector = DragonConnector()

        try:
            # connect executes the complete handshake and raises an exception if comms fails
            connector.connect_to_dragon()
        finally:
            connector.cleanup()


def test_load_env_no_file(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Ensure an empty dragon .env file doesn't break the launcher"""
    test_path = pathlib.Path(test_dir)
    # mock_dragon_root = pathlib.Path(test_dir) / "dragon"
    # exp_env_path = pathlib.Path(test_dir) / "dragon" / ".env"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)

        dragon_conf = smartsim._core.config.CONFIG.dragon_dotenv
        # verify config doesn't exist
        assert not dragon_conf.exists()

        connector = DragonConnector()

        loaded_env = connector.load_persisted_env()
        assert not loaded_env


def test_load_env_env_file_created(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Ensure a populated dragon .env file is loaded correctly by the launcher"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)
        create_dotenv(mock_dragon_root, DEFAULT_DRAGON_VERSION)
        dragon_conf = smartsim._core.config.CONFIG.dragon_dotenv

        # verify config does exist
        assert dragon_conf.exists()

        # load config w/launcher
        connector = DragonConnector()

        loaded_env = connector.load_persisted_env()
        assert loaded_env

        # confirm .env was parsed as expected by inspecting a key
        assert "DRAGON_BASE_DIR" in loaded_env
        base_dir = loaded_env["DRAGON_BASE_DIR"]

        assert "DRAGON_ROOT_DIR" in loaded_env
        assert loaded_env["DRAGON_ROOT_DIR"] == base_dir

        assert "DRAGON_INCLUDE_DIR" in loaded_env
        assert loaded_env["DRAGON_INCLUDE_DIR"] == f"{base_dir}/include"

        assert "DRAGON_LIB_DIR" in loaded_env
        assert loaded_env["DRAGON_LIB_DIR"] == f"{base_dir}/lib"

        assert "DRAGON_VERSION" in loaded_env
        assert loaded_env["DRAGON_VERSION"] == DEFAULT_DRAGON_VERSION

        assert "PATH" in loaded_env
        assert loaded_env["PATH"] == f"{base_dir}/bin"

        assert "LD_LIBRARY_PATH" in loaded_env
        assert loaded_env["LD_LIBRARY_PATH"] == f"{base_dir}/lib"


def test_load_env_cached_env(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Ensure repeated attempts to use dragon env don't hit file system"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)
        create_dotenv(mock_dragon_root, DEFAULT_DRAGON_VERSION)

        # load config w/launcher
        connector = DragonConnector()

        loaded_env = connector.load_persisted_env()
        assert loaded_env

        # ensure attempting to reload would bomb
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", None)

        # attempt to load and if it doesn't blow up, it used the cached copy

        loaded_env = connector.load_persisted_env()
        assert loaded_env


def test_merge_env(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Ensure that merging dragon .env file into current env has correct precedences"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)
        create_dotenv(mock_dragon_root, DEFAULT_DRAGON_VERSION)

        # load config w/launcher
        connector = DragonConnector()
        loaded_env = {**connector.load_persisted_env()}
        assert loaded_env

        curr_base_dir = "/foo"
        curr_path = "/foo:/bar"
        curr_only = "some-value"

        loaded_path = loaded_env.get("PATH", "")

        # ensure some non-dragon value exists in env; we want
        # to see that it is in merged output without empty prepending
        non_dragon_key = "NON_DRAGON_KEY"
        non_dragon_value = "non_dragon_value"
        connector._env_vars[non_dragon_key] = non_dragon_value

        curr_env = {
            "DRAGON_BASE_DIR": curr_base_dir,  # expect overwrite
            "PATH": curr_path,  # expect prepend
            "ONLY_IN_CURRENT": curr_only,  # expect pass-through
        }

        merged_env = connector.merge_persisted_env(curr_env)

        # any dragon env vars should be overwritten
        assert merged_env["DRAGON_BASE_DIR"] != curr_base_dir

        # any non-dragon collisions should result in prepending
        assert merged_env["PATH"] == f"{loaded_path}:{curr_path}"
        # ensure we actually see a change
        assert merged_env["PATH"] != loaded_env["PATH"]

        # any keys that were in curr env should still exist, unchanged
        assert merged_env["ONLY_IN_CURRENT"] == curr_only

        # any non-dragon keys that didn't exist avoid unnecessary prepending
        assert merged_env[non_dragon_key] == non_dragon_value


def test_run_step_fail(test_dir: str) -> None:
    """Verify that the dragon launcher still returns the step id
    when the running step fails"""
    test_path = pathlib.Path(test_dir)
    status_dir = (test_path / ".smartsim" / "logs").as_posix()

    rs = DragonRunSettings(exe="sleep", exe_args=["1"])
    step0 = DragonStep("step0", test_dir, rs)
    step0.meta["status_dir"] = status_dir

    mock_connector = MagicMock(spec=DragonConnector)
    mock_connector.is_connected = True
    mock_connector.send_request = MagicMock(
        return_value=DragonRunResponse(step_id=step0.name, error_message="mock fail!")
    )
    mock_connector.merge_persisted_env = MagicMock(
        return_value={"FOO": "bar", "BAZ": "boop"}
    )

    launcher = DragonLauncher()
    launcher._connector = mock_connector

    result = launcher.run(step0)

    # verify the failed step name is in the result
    assert step0.name in result


def test_run_step_batch_empty(dragon_batch_step: DragonBatchStep) -> None:
    """Verify that the dragon launcher behaves when asked to execute
    a batch step that has no sub-steps"""
    # remove the steps added in the batch fixture
    dragon_batch_step.steps.clear()

    mock_step_id = "MOCK-STEPID"
    mock_connector = MagicMock()  # DragonConnector()
    mock_connector.is_connected = True
    mock_connector.send_request = MagicMock(
        return_value=DragonRunResponse(
            step_id=dragon_batch_step.name, error_message="mock fail!"
        )
    )

    launcher = DragonLauncher()
    launcher._connector = mock_connector
    launcher.task_manager.start_and_wait = MagicMock(return_value=(0, mock_step_id, ""))

    result = launcher.run(dragon_batch_step)

    # verify a step name is returned
    assert result
    # verify the batch step name is not in the result (renamed to SLURM-*)
    assert dragon_batch_step.name not in result

    send_invocation = mock_connector.send_request

    # verify a batch request is not sent through the dragon connector
    send_invocation.assert_not_called()


def test_run_step_batch_failure(dragon_batch_step: DragonBatchStep) -> None:
    """Verify that the dragon launcher sends returns the step id
    when the running step fails"""
    mock_connector = MagicMock()  # DragonConnector()
    mock_connector.is_connected = True
    mock_connector.send_request = MagicMock(
        return_value=DragonRunResponse(
            step_id=dragon_batch_step.name, error_message="mock fail!"
        )
    )

    mock_step_id = "MOCK-STEPID"
    error_msg = "DOES_NOT_COMPUTE!"
    launcher = DragonLauncher()
    launcher._connector = mock_connector
    launcher.task_manager.start_and_wait = MagicMock(
        return_value=(1, mock_step_id, error_msg)
    )

    # a non-zero return code from the batch script should raise an error
    with pytest.raises(LauncherError) as ex:
        launcher.run(dragon_batch_step)

    # verify the correct error message is in the exception
    assert error_msg in ex.value.args[0]


def test_run_step_success(test_dir: str) -> None:
    """Verify that the dragon launcher sends the correctly formatted request for a step"""
    test_path = pathlib.Path(test_dir)
    status_dir = (test_path / ".smartsim" / "logs").as_posix()

    rs = DragonRunSettings(exe="sleep", exe_args=["1"])
    step0 = DragonStep("step0", test_dir, rs)
    step0.meta["status_dir"] = status_dir

    mock_connector = MagicMock(spec=DragonConnector)
    mock_connector.is_connected = True
    mock_connector.send_request = MagicMock(
        return_value=DragonRunResponse(step_id=step0.name)
    )

    launcher = DragonLauncher()
    launcher._connector = mock_connector
    mock_connector.merge_persisted_env = MagicMock(
        return_value={"FOO": "bar", "BAZ": "boop"}
    )

    result = launcher.run(step0)

    # verify the successfully executed step name is in the result
    assert step0.name in result

    # verify the DragonRunRequest sent matches all expectations
    send_invocation = mock_connector.send_request
    send_invocation.assert_called_once()

    args = send_invocation.call_args[0]  # call_args == t.Tuple[args, kwargs]

    dragon_run_request = args[0]
    req_name = dragon_run_request.name  # name sent to dragon env
    assert req_name.startswith(step0.name)

    req_policy_cpu_affinity = dragon_run_request.policy.cpu_affinity
    assert not req_policy_cpu_affinity  # default should be empty list

    req_policy_gpu_affinity = dragon_run_request.policy.gpu_affinity
    assert not req_policy_gpu_affinity  # default should be empty list


def test_run_step_success_batch(
    monkeypatch: pytest.MonkeyPatch, dragon_batch_step: DragonBatchStep
) -> None:
    """Verify that the dragon launcher sends the correctly formatted request
    for a batch step"""
    mock_connector = MagicMock()  # DragonConnector()
    mock_connector.is_connected = True
    mock_connector.send_request = MagicMock(
        return_value=DragonRunResponse(step_id=dragon_batch_step.name)
    )

    launcher = DragonLauncher()
    launcher._connector = mock_connector
    launcher.task_manager.start_and_wait = MagicMock(return_value=(0, "success", ""))

    result = launcher.run(dragon_batch_step)

    # verify the successfully executed step name is in the result
    assert dragon_batch_step.name not in result
    assert result

    send_invocation = mock_connector.send_request

    # verify a batch request is not sent through the dragon connector
    send_invocation.assert_not_called()
    launcher.task_manager.start_and_wait.assert_called_once()

    args = launcher.task_manager.start_and_wait.call_args[0]

    # verify the batch script is executed
    launch_cmd = dragon_batch_step.get_launch_cmd()
    for stmt in launch_cmd:
        assert stmt in args[0]  # args[0] is the cmd list sent to subprocess.Popen
