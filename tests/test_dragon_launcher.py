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

import typing as t

import pytest

from smartsim._core.launcher.dragon.dragonLauncher import DragonLauncher
from smartsim._core.schemas.dragonRequests import DragonBootstrapRequest
from smartsim.error.errors import LauncherError

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


class MockPopen:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None: ...

    @property
    def pid(self) -> int:
        return 1

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


def test_dragon_connect_bind_address(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Test the connection to a dragon environment dynamically selects an open port
    in the range supplied"""

    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonLauncher.get_best_interface_and_address",
            lambda: ("faux_interface", "127.0.0.1"),
        )
        ctx.setattr(
            "smartsim._core.launcher.dragon.dragonLauncher.DragonLauncher._handshake",
            lambda self, address: ...,
        )

        mock_socket = MockSocket()

        ctx.setattr("zmq.Context.socket", mock_socket)
        ctx.setattr("subprocess.Popen", lambda *args, **kwargs: MockPopen())

        dragon_launcher = DragonLauncher()
        with pytest.raises(LauncherError) as ex:
            # it will complain about failure to connect when validating...
            dragon_launcher.connect_to_dragon(test_dir)

        chosen_port = int(mock_socket.bind_address.split(":")[-1])
        assert chosen_port >= 5995
