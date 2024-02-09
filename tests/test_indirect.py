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


import pathlib
import sys

import psutil
import pytest

from smartsim._core.config import CONFIG
from smartsim._core.entrypoints.indirect import cleanup, get_parser, get_ts, main
from smartsim._core.utils.helpers import encode_cmd

ALL_ARGS = {
    "+command",
    "+entity_type",
    "+telemetry_dir",
    "+output_file",
    "+error_file",
    "+working_dir",
}

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


# fmt: off
@pytest.mark.parametrize(
        ["cmd", "missing"],
        [
            pytest.param("indirect.py", {"+name", "+command", "+entity_type", "+telemetry_dir", "+working_dir"}, id="no args"),
            pytest.param("indirect.py -c echo +entity_type ttt +telemetry_dir ddd +output_file ooo +working_dir www +error_file eee", {"+command"}, id="cmd typo"),
            pytest.param("indirect.py -t orchestrator +command ccc +telemetry_dir ddd +output_file ooo +working_dir www +error_file eee", {"+entity_type"}, id="etype typo"),
            pytest.param("indirect.py -d /foo/bar +entity_type ttt +command ccc +output_file ooo +working_dir www +error_file eee", {"+telemetry_dir"}, id="dir typo"),
            pytest.param("indirect.py        +entity_type ttt +telemetry_dir ddd +output_file ooo +working_dir www +error_file eee", {"+command"}, id="no cmd"),
            pytest.param("indirect.py +command ccc        +telemetry_dir ddd +output_file ooo +working_dir www +error_file eee", {"+entity_type"}, id="no etype"),
            pytest.param("indirect.py +command ccc +entity_type ttt        +output_file ooo +working_dir www +error_file eee", {"+telemetry_dir"}, id="no dir"),
        ]
)
# fmt: on
def test_parser(capsys, cmd, missing):
    """Test that the parser reports any missing required arguments"""
    parser = get_parser()

    args = cmd.split()

    captured = capsys.readouterr()  # throw away existing output
    with pytest.raises(SystemExit) as ex:
        ns = parser.parse_args(args)

    captured = capsys.readouterr()
    assert "the following arguments are required" in captured.err
    for arg in missing:
        assert arg in captured.err

    expected = ALL_ARGS - missing
    msg_tuple = captured.err.split("the following arguments are required: ")
    if len(msg_tuple) < 2:
        assert False, "error message indicates no missing arguments"

    actual_missing = msg_tuple[1].strip()
    for exp in expected:
        assert f"{exp}/" not in actual_missing


def test_cleanup(capsys, monkeypatch):
    """Ensure cleanup attempts termination of correct process"""
    mock_pid = 123
    create_msg = "creating: {0}"
    term_msg = "terminating: {0}"

    class MockProc:
        def __init__(self, pid: int):
            print(create_msg.format(pid))

        def terminate(self):
            print(term_msg.format(mock_pid))

    captured = capsys.readouterr()  # throw away existing output

    with monkeypatch.context() as ctx:
        ctx.setattr("psutil.pid_exists", lambda pid: True)
        ctx.setattr("psutil.Process", MockProc)
        ctx.setattr("smartsim._core.entrypoints.indirect.STEP_PID", mock_pid)
        cleanup()

    captured = capsys.readouterr()
    assert create_msg.format(mock_pid) in captured.out
    assert term_msg.format(mock_pid) in captured.out


def test_cleanup_late(capsys, monkeypatch):
    """Ensure cleanup exceptions are swallowed if a process is already terminated"""
    mock_pid = 123
    create_msg = "creating: {0}"
    term_msg = "terminating: {0}"

    class MockMissingProc:
        def __init__(self, pid: int) -> None:
            print(create_msg.format(mock_pid))
            raise psutil.NoSuchProcess(pid)

        def terminate(self) -> None:
            print(term_msg.format(mock_pid))

    captured = capsys.readouterr()  # throw away existing output

    with monkeypatch.context() as ctx:
        ctx.setattr("psutil.pid_exists", lambda pid: True)
        ctx.setattr("psutil.Process", MockMissingProc)
        ctx.setattr("smartsim._core.entrypoints.indirect.STEP_PID", mock_pid)
        cleanup()

    captured = capsys.readouterr()
    assert create_msg.format(mock_pid) in captured.out


def test_ts():
    """Ensure expected output type"""
    ts = get_ts()
    assert isinstance(ts, int)


def test_indirect_main_dir_check(test_dir):
    """Ensure that the proxy validates the test directory exists"""
    exp_dir = pathlib.Path(test_dir)

    cmd = ["echo", "unit-test"]
    encoded_cmd = encode_cmd(cmd)

    status_path = exp_dir / CONFIG.telemetry_subdir

    # show that a missing status_path is created when missing
    main(encoded_cmd, "application", exp_dir, status_path)

    assert status_path.exists()


def test_indirect_main_cmd_check(capsys, test_dir, monkeypatch):
    """Ensure that the proxy validates the cmd is not empty or whitespace-only"""
    exp_dir = pathlib.Path(test_dir)

    captured = capsys.readouterr()  # throw away existing output
    with monkeypatch.context() as ctx, pytest.raises(ValueError) as ex:
        ctx.setattr("smartsim._core.entrypoints.indirect.logger.error", print)
        _ = main("", "application", exp_dir, exp_dir / CONFIG.telemetry_subdir)

    captured = capsys.readouterr()
    assert "Invalid cmd supplied" in ex.value.args[0]

    # test with non-emptystring cmd
    with monkeypatch.context() as ctx, pytest.raises(ValueError) as ex:
        ctx.setattr("smartsim._core.entrypoints.indirect.logger.error", print)
        status_dir = exp_dir / CONFIG.telemetry_subdir
        _ = main("  \n  \t   ", "application", exp_dir, status_dir)

    captured = capsys.readouterr()
    assert "Invalid cmd supplied" in ex.value.args[0]


def test_complete_process(fileutils, test_dir):
    """Ensure the happy-path completes and returns a success return code"""
    script = fileutils.get_test_conf_path("sleep.py")

    exp_dir = pathlib.Path(test_dir)

    raw_cmd = f"{sys.executable} {script} --time=1"
    cmd = encode_cmd(raw_cmd.split())

    rc = main(cmd, "application", exp_dir, exp_dir / CONFIG.telemetry_subdir)
    assert rc == 0

    assert exp_dir.exists()

    # NOTE: don't have a manifest so we're falling back to default event path
    data_dir = exp_dir / CONFIG.telemetry_subdir
    start_events = list(data_dir.rglob("start.json"))
    stop_events = list(data_dir.rglob("stop.json"))

    assert start_events
    assert stop_events
