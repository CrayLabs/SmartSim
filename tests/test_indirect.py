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


import pathlib
import psutil
import pytest
import sys
import uuid

from smartsim._core.entrypoints.indirect import get_parser, cleanup, get_ts, main


ALL_ARGS = {"+c", "+t", "+n", "+d"}


@pytest.mark.parametrize(
        ["cmd", "missing"],
        [
            pytest.param("indirect.py", {"+c", "+t", "+n", "+d"}, id="no args"),
            pytest.param("indirect.py -c echo", {"+c", "+t", "+n", "+d"}, id="cmd typo"),
            pytest.param("indirect.py -t orchestrator", {"+c", "+t", "+n", "+d"}, id="etype typo"),
            pytest.param("indirect.py -n expname", {"+c", "+t", "+n", "+d"}, id="name typo"),
            pytest.param("indirect.py -d /foo/bar", {"+c", "+t", "+n", "+d"}, id="dir typo"),
            pytest.param("indirect.py        +t ttt +d ddd +n nnn", {"+c", "+t", "+n", "+d"}, id="no cmd"),
            pytest.param("indirect.py +c ccc        +d ddd +n nnn", {"+c", "+t", "+n", "+d"}, id="no etype"),
            pytest.param("indirect.py +c ccc +t ttt +d ddd       ", {"+c", "+t", "+n", "+d"}, id="no name"),
            pytest.param("indirect.py +c ccc +t ttt        +n nnn", {"+c", "+t", "+n", "+d"}, id="no dir"),
        ]
)
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
    for exp in expected:
        assert exp not in captured.err


def test_cleanup(capsys, monkeypatch):
    """Ensure cleanup attempts termination of correcct process"""
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
        ctx.setattr('psutil.Process', MockProc)
        ctx.setattr('smartsim._core.entrypoints.indirect.STEP_PID', mock_pid)
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
        def __init__(self, pid: int):
            print(create_msg.format(mock_pid))
            raise psutil.NoSuchProcess(pid)
        def terminate(self):
            print(term_msg.format(mock_pid))
    
    captured = capsys.readouterr()  # throw away existing output

    with monkeypatch.context() as ctx:
        ctx.setattr('psutil.Process', MockMissingProc)
        ctx.setattr('smartsim._core.entrypoints.indirect.STEP_PID', mock_pid)
        cleanup()

    captured = capsys.readouterr()
    assert create_msg.format(mock_pid) in captured.out


def test_ts():
    """Ensure expected output type"""
    ts = get_ts()
    assert isinstance(ts, int)


def test_indirect_main_dir_check():
    """Ensure that the proxy validates the test directory exists"""
    test_dir = f"/foo/{uuid.uuid4()}"
    exp_dir = pathlib.Path(test_dir)
    std_out = str(exp_dir / "out.txt")
    err_out = str(exp_dir / "err.txt")

    with pytest.raises(ValueError) as ex:
        main("echo unit-test", "application", "unit-test-step-1", std_out, err_out, exp_dir)

    assert "directory does not exist" in ex.value.args[0]


def test_indirect_main_cmd_check(capsys, fileutils, monkeypatch):
    """Ensure that the proxy validates the cmd is not empty or whitespace-only"""
    test_dir = fileutils.make_test_dir()
    exp_dir = pathlib.Path(test_dir)
    std_out = str(exp_dir / "out.txt")
    err_out = str(exp_dir / "err.txt")

    captured = capsys.readouterr()  # throw away existing output
    with monkeypatch.context() as ctx, pytest.raises(ValueError) as ex:
        ctx.setattr('smartsim._core.entrypoints.indirect.logger.error', print)
        _ = main("", "application", "unit-test-step-1", std_out, err_out, exp_dir)

    captured = capsys.readouterr()
    assert "Invalid cmd supplied" in ex.value.args[0]

    std_out = str(exp_dir / "out.txt")
    err_out = str(exp_dir / "err.txt")

    # test with non-emptystring cmd
    with monkeypatch.context() as ctx, pytest.raises(ValueError) as ex:
        ctx.setattr('smartsim._core.entrypoints.indirect.logger.error', print)
        _ = main("  \n  \t   ", "application", "unit-test-step-1", std_out, err_out, exp_dir)

    captured = capsys.readouterr()
    assert "Invalid cmd supplied" in ex.value.args[0]


def test_complete_process(capsys, fileutils):
    """Ensure the happy-path completes and returns a success return code"""
    script = fileutils.get_test_conf_path("sleep.py")

    test_dir = fileutils.make_test_dir()
    exp_dir = pathlib.Path(test_dir)
    std_out = str(exp_dir / "out.txt")
    err_out = str(exp_dir / "err.txt")

    captured = capsys.readouterr()  # throw away existing output
    rc = main(f"{sys.executable} {script} --time=1", "application", "unit-test-step-1", std_out, err_out, exp_dir)
    assert rc == 0

    app_dir = exp_dir / "manifest" / "application" / "unit-test-step-1"
    assert app_dir.exists()
    
    start_evt = app_dir / "start.json"
    exit_evt = app_dir / "stop.json"

    assert start_evt.exists()
    assert start_evt.is_file()

    assert exit_evt.exists()
    assert exit_evt.is_file()
