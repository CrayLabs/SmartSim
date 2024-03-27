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

    # def recv_json(self) -> str:
    #     dbr = DragonBootstrapRequest(address=self._bind_address)
    #     return dbr.json()

    def recv_string(self) -> str:
        dbr = DragonBootstrapRequest(address=self._bind_address)
        return f"bootstrap|{dbr.json()}"

    def close(self) -> None: ...

    def send_json(self, json: str) -> None: ...

    def send_string(*args, **kwargs) -> None: ...

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
