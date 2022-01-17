import psutil
import pytest

from smartsim._core.launcher.util.shell import *


def test_execute_cmd():
    returncode, out, err = execute_cmd(["hostname"])

    assert isinstance(returncode, int)
    assert isinstance(out, str)
    assert isinstance(err, str)


def test_execute_async_cmd():
    proc = execute_async_cmd(["hostname"], cwd=".")
    proc.communicate()

    assert isinstance(proc, psutil.Popen)


def test_errors():
    with pytest.raises(ShellError):
        execute_async_cmd(["sleep", "--notexistingoption"], cwd=".")

    with pytest.raises(ShellError):
        execute_cmd(["sleep", "3"], timeout=1)
