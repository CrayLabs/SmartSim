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
import os
import pathlib
import typing as t

import pytest

# retrieved from pytest fixtures
if pytest.test_launcher != "dragon":
    pytestmark = pytest.mark.skip(reason="Test is only for Dragon WLM systems")

try:
    import smartsim._core.entrypoints.dragon as drg
except:
    pytest.skip("Unable to import Dragon library", allow_module_level=True)


@pytest.fixture
def mock_argv() -> t.List[str]:
    """Fixture for returning valid arguments to the entrypoint"""
    return ["+launching_address", "mock-addr", "+interface", "mock-interface"]


def test_file_removal(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that the log file is removed when expected"""
    mock_file_name = "mocked_file_name.txt"
    expected_path = pathlib.Path(test_dir) / mock_file_name
    expected_path.touch()

    with monkeypatch.context() as ctx:
        # ensure we get outputs in the test directory
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.get_log_path", lambda: str(expected_path)
        )

        drg.remove_config_log()
        assert not expected_path.exists(), "Dragon config file was not removed"


def test_file_removal_on_bad_path(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that file removal doesn't blow up if the log file wasn't created"""
    mock_file_name = "mocked_file_name.txt"
    expected_path = pathlib.Path(test_dir) / mock_file_name

    with monkeypatch.context() as ctx:
        # ensure we get outputs in the test directory
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.get_log_path", lambda: str(expected_path)
        )

        # confirm the file doesn't exist...
        assert not expected_path.exists(), "Dragon config file was not found"

        try:
            # ensure we don't blow up
            drg.remove_config_log()
        except:
            assert False


def test_dragon_failure(
    mock_argv: t.List[str], test_dir: str, monkeypatch: pytest.MonkeyPatch
):
    """Verify that the expected cleanup actions are taken when the dragon
    entrypoint exits"""
    mock_file_name = "mocked_file_name.txt"
    expected_path = pathlib.Path(test_dir) / mock_file_name
    expected_path.touch()

    with monkeypatch.context() as ctx:
        # ensure we get outputs in the test directory
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.get_log_path", lambda: str(expected_path)
        )

        def raiser(args_) -> int:
            raise Exception("Something bad...")

        # we don't need to execute the entrypoint...
        ctx.setattr("smartsim._core.entrypoints.dragon.execute_entrypoint", raiser)

        return_code = drg.main(mock_argv)

        # ensure our exception error code is returned
        assert return_code == -1


def test_dragon_main(
    mock_argv: t.List[str], test_dir: str, monkeypatch: pytest.MonkeyPatch
):
    """Verify that the expected startup & cleanup actions are taken when the dragon
    entrypoint exits"""
    mock_file_name = "mocked_file_name.txt"
    expected_path = pathlib.Path(test_dir) / mock_file_name
    expected_path.touch()

    with monkeypatch.context() as ctx:
        # ensure we get outputs in the test directory
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.get_log_path", lambda: str(expected_path)
        )
        # we don't need to execute the actual entrypoint...
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.execute_entrypoint", lambda args_: 0
        )

        return_code = drg.main(mock_argv)

        # execute_entrypoint should return 0 from our mock
        assert return_code == 0
        # the cleanup should remove our config file
        assert not expected_path.exists(), "Dragon config file was not removed!"
        # the environment should be set as expected
        assert os.environ.get("PYTHONUNBUFFERED", None) == "1"


@pytest.mark.parametrize(
    "signal_num",
    [
        pytest.param(0, id="non-truthy signal"),
        pytest.param(-1, id="negative signal"),
        pytest.param(1, id="positive signal"),
    ],
)
def test_signal_handler(signal_num: int, monkeypatch: pytest.MonkeyPatch):
    """Verify that the signal handler performs expected actions"""
    counter: int = 0

    def increment_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    with monkeypatch.context() as ctx:
        ctx.setattr("smartsim._core.entrypoints.dragon.cleanup", increment_counter)
        ctx.setattr("smartsim._core.entrypoints.dragon.logger.info", increment_counter)

        drg.handle_signal(signal_num, None)

        # show that we log informational message & do cleanup (take 2 actions)
        assert counter == 2


def test_log_path(monkeypatch: pytest.MonkeyPatch):
    """Verify that the log path is loaded & returned as expected"""

    with monkeypatch.context() as ctx:
        expected_filename = "foo.log"
        ctx.setattr(
            "smartsim._core.config.config.Config.dragon_log_filename", expected_filename
        )

        log_path = drg.get_log_path()

        assert expected_filename in log_path


def test_summary(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that the summary is written to expected location w/expected information"""

    with monkeypatch.context() as ctx:
        expected_ip = "127.0.0.111"
        expected_interface = "mock_int0"
        summary_file = pathlib.Path(test_dir) / "foo.log"
        expected_hostname = "mockhostname"

        ctx.setattr(
            "smartsim._core.config.config.Config.dragon_log_filename",
            str(summary_file),
        )
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.socket.gethostname",
            lambda: expected_hostname,
        )

        drg.print_summary(expected_interface, expected_ip)

        summary = summary_file.read_text()

        assert expected_ip in summary
        assert expected_interface in summary
        assert expected_hostname in summary


def test_cleanup(monkeypatch: pytest.MonkeyPatch):
    """Verify that the cleanup function attempts to remove the log file"""
    counter: int = 0

    def increment_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    with monkeypatch.context() as ctx:
        ctx.setattr(
            "smartsim._core.entrypoints.dragon.remove_config_log", increment_counter
        )
        drg.SHUTDOWN_INITIATED = False
        drg.cleanup()

        # show that cleanup removes config
        assert counter == 1
        # show that cleanup alters the flag to enable shutdown
        assert drg.SHUTDOWN_INITIATED


def test_signal_handler_registration(test_dir: str, monkeypatch: pytest.MonkeyPatch):
    """Verify that signal handlers are registered for all expected signals"""
    sig_nums: t.List[int] = []

    def track_args(*args, **kwargs):
        nonlocal sig_nums
        sig_nums.append(args[0])

    with monkeypatch.context() as ctx:
        ctx.setattr("smartsim._core.entrypoints.dragon.signal.signal", track_args)

        # ensure valid start point
        assert not sig_nums

        drg.register_signal_handlers()

        # ensure all expected handlers are registered
        assert set(sig_nums) == set(drg.SIGNALS)


def test_arg_parser__no_args():
    """Verify arg parser fails when no args are not supplied"""
    args_list = []

    with pytest.raises(SystemExit) as ex:
        # ensure that parser complains about missing required arguments
        drg.parse_arguments(args_list)


def test_arg_parser__invalid_launch_addr():
    """Verify arg parser fails with empty launch_address"""
    addr_flag = "+launching_address"
    addr_value = ""

    args_list = [addr_flag, addr_value]

    with pytest.raises(ValueError) as ex:
        args = drg.parse_arguments(args_list)


def test_arg_parser__required_only():
    """Verify arg parser succeeds when optional args are omitted"""
    addr_flag = "+launching_address"
    addr_value = "mock-address"

    args_list = [addr_flag, addr_value]

    args = drg.parse_arguments(args_list)

    assert args.launching_address == addr_value
    assert not args.interface


def test_arg_parser__with_optionals():
    """Verify arg parser succeeds when optional args are included"""
    addr_flag = "+launching_address"
    addr_value = "mock-address"

    interface_flag = "+interface"
    interface_value = "mock-int"

    args_list = [interface_flag, interface_value, addr_flag, addr_value]

    args = drg.parse_arguments(args_list)

    assert args.launching_address == addr_value
    assert args.interface == interface_value
