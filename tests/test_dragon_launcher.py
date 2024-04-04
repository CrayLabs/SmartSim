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

import multiprocessing as mp
import os
import typing as t

import pytest
import zmq

from smartsim._core.config.config import get_config
from smartsim._core.launcher.dragon.dragonLauncher import DragonLauncher
from smartsim._core.launcher.dragon.dragonSockets import get_secure_socket
from smartsim._core.schemas.dragonRequests import DragonBootstrapRequest
from smartsim._core.schemas.dragonResponses import DragonHandshakeResponse
from smartsim._core.utils.network import IFConfig, find_free_port
from smartsim._core.utils.security import KeyManager

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


class MockPopen:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None: ...

    @property
    def pid(self) -> int:
        return 99999

    @property
    def returncode(self) -> int:
        return 0


class MockSocket:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self._bind_address = ""

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self

    def bind(self, addr: str) -> None:
        self._bind_address = addr

    def recv_string(self) -> str:
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
    def __init__(self, context: zmq.Context) -> None:
        self.num_starts: int = 0
        self.num_stops: int = 0
        self.num_configure_curves: int = 0
        self.context = context

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
    try:
        context = zmq.Context()
        addr = "127.0.0.1"
        callback_port = kwargs["port"]
        head_port = find_free_port(start=callback_port + 1)

        callback_socket, dragon_authenticator = get_secure_socket(
            context, zmq.REQ, False
        )

        dragon_head_socket, dragon_authenticator = get_secure_socket(
            context, zmq.REP, True, dragon_authenticator
        )

        full_addr = f"{addr}:{callback_port}"
        callback_socket.connect(f"tcp://{full_addr}")

        full_head_addr = f"tcp://{addr}:{head_port}"
        dragon_head_socket.bind(full_head_addr)

        req = DragonBootstrapRequest(address=full_head_addr)

        msg_sent = False
        while not msg_sent:
            callback_socket.send_string("bootstrap|" + req.json())
            # hold until bootstrap response is received
            _ = callback_socket.recv()
            msg_sent = True

        hand_shaken = False
        while not hand_shaken:
            # other side should set up a socket and push me a `HandshakeRequest`
            _ = dragon_head_socket.recv()
            # acknowledge handshake success w/DragonHandshakeResponse
            handshake_ack = DragonHandshakeResponse(dragon_pid=os.getpid())
            dragon_head_socket.send_string(f"handshake|{handshake_ack.json()}")

            hand_shaken = True
    except Exception as ex:
        print(f"exception occurred while configuring mock handshaker: {ex}")
    finally:
        dragon_authenticator.stop()
        callback_socket.close()
        dragon_head_socket.close()


def test_dragon_connect_bind_address(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Test the connection to a dragon environment dynamically selects an open port
    in the range supplied"""

    with monkeypatch.context() as ctx:
        # make sure we don't touch "real keys" during a test
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        mock_socket = MockSocket()

        # look at test_dir for dragon config
        ctx.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
        # avoid finding real interface
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonLauncher.get_best_interface_and_address",
            lambda: IFConfig(interface="faux_interface", address="127.0.0.1"),
        )
        # we need to set the socket value or is_connected returns False
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonLauncher.DragonLauncher._handshake",
            lambda self, address: setattr(self, "_dragon_head_socket", mock_socket),
        )
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)
        # avoid starting a real zmq socket
        ctx.setattr("zmq.Context.socket", mock_socket)
        # avoid starting a real process for dragon entrypoint
        ctx.setattr("subprocess.Popen", lambda *args, **kwargs: MockPopen())

        dragon_launcher = DragonLauncher()
        dragon_launcher.connect_to_dragon(test_dir)

        chosen_port = int(mock_socket.bind_address.split(":")[-1])
        assert chosen_port >= 5995


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
    context = zmq.Context()

    with monkeypatch.context() as ctx:
        # look at test dir for dragon config
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)

        _, authenticator = get_secure_socket(context, socket_type, is_server=is_server)

        km = KeyManager(get_config(), as_server=is_server)

        assert isinstance(authenticator, MockAuthenticator)

        # ensure authenticator was configured
        assert authenticator.num_configure_curves > 0
        # ensure authenticator was started
        assert authenticator.num_starts > 0
        assert authenticator.context == context
        # ensure authenticator will accept any secured connection
        assert authenticator.cfg_kwargs.get("domain", "") == "*"
        # ensure authenticator is using the expected set of keys
        assert authenticator.cfg_kwargs.get("location", "") == km.client_keys_dir


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
    context = zmq.Context()

    with monkeypatch.context() as ctx:
        # look at test dir for dragon config
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)
        # avoid starting a real authenticator thread
        ctx.setattr("zmq.auth.thread.ThreadAuthenticator", MockAuthenticator)

        socket, _ = get_secure_socket(context, zmq.REP, as_server)

        # verify the socket is correctly configured to use curve authentication
        assert bool(socket.CURVE_SERVER) == as_server
        assert not socket.closed

        socket.close()


def test_secure_socket(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Ensure the authenticator created by the secure socket factory method
    is fully configured and started when returned to a client"""

    with monkeypatch.context() as ctx:
        # make sure we don't touch "real keys" during a test
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        context = zmq.Context()
        server, authenticator = get_secure_socket(context, zmq.REP, True)

        ip, port = "127.0.0.1", find_free_port(start=9999)

        try:
            server.bind(f"tcp://*:{port}")

            client, authenticator = get_secure_socket(
                context, zmq.REQ, False, authenticator
            )

            client.connect(f"tcp://{ip}:{port}")

            to_send = "you get a foo! you get a foo! everybody gets a foo!"
            client.send_string(to_send, flags=zmq.NOBLOCK)

            received_msg = server.recv_string()
            assert received_msg == to_send
            print("server receieved: ", received_msg)
        finally:
            if authenticator:
                authenticator.stop()
            if client:
                client.close()
            if server:
                server.close()


# def test_dragon_launcher_handshake(monkeypatch: pytest.MonkeyPatch, test_dir: str):
#     """Test that a real handshake between a launcher & dragon environment
#     completes successfully using secure sockets"""
#     context = zmq.Context()
#     addr = "127.0.0.1"
#     bootstrap_port = find_free_port(start=5995)

#     with monkeypatch.context() as ctx:
#         # make sure we don't touch "real keys" during a test
#         ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

#         # look at test dir for dragon config
#         ctx.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
#         # avoid finding real interface since we may not be on a super
#         ctx.setattr(
#             "smartsim._core.launcher.dragon.dragonLauncher.get_best_interface_and_address",
#             lambda: IFConfig("faux_interface", addr),
#         )

#         # start up a faux dragon env that knows how to do the handshake process
#         # but uses secure sockets for all communication.
#         mock_dragon = mp.Process(
#             target=mock_dragon_env,
#             daemon=True,
#             kwargs={"port": bootstrap_port, "test_dir": test_dir},
#         )

#         def fn(*args, **kwargs):
#             mock_dragon.start()
#             return mock_dragon

#         ctx.setattr("subprocess.Popen", fn)

#         launcher = DragonLauncher()

#         try:
#             # connect executes the complete handshake and raises an exception if comms fails
#             launcher.connect_to_dragon(test_dir)
#         finally:
#             launcher.cleanup()
