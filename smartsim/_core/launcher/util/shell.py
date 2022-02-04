import time
import psutil

from subprocess import PIPE, TimeoutExpired

from ...utils.helpers import check_dev_log_level
from ....error import ShellError
from ....log import get_logger

logger = get_logger(__name__)
verbose_shell = check_dev_log_level()


def execute_cmd(cmd_list, shell=False, cwd=None, env=None, proc_input="", timeout=None):
    """Execute a command locally

    :param cmd_list: list of command with arguments
    :type cmd_list: list of str
    :param shell: run in system shell, defaults to False
    :type shell: bool, optional
    :param cwd: current working directory, defaults to None
    :type cwd: str, optional
    :param env: environment to launcher process with,
                defaults to None (current env)
    :type env: dict, optional
    :param proc_input: input to the process, defaults to ""
    :type proc_input: str, optional
    :param timeout: timeout of the process, defaults to None
    :type timeout: int, optional
    :raises ShellError: if timeout of process was exceeded
    :raises ShellError: if child process raises an error
    :return: returncode, output, and error of the process
    :rtype: tuple of (int, str, str)
    """
    global verbose_shell

    if verbose_shell:
        source = "shell" if shell else "Popen"
        logger.debug(f"Executing {source} cmd: {' '.join(cmd_list)}")

    # spawning the subprocess and connecting to its output
    proc = psutil.Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=cwd, shell=shell, env=env
    )
    try:
        proc_input = proc_input.encode("utf-8")
        out, err = proc.communicate(input=proc_input, timeout=timeout)
    except TimeoutExpired as e:
        proc.kill()
        _, errs = proc.communicate()
        logger.error(errs)
        raise ShellError(
            "Failed to execute command, timeout reached", e, cmd_list
        ) from None
    except OSError as e:
        raise ShellError(
            "Exception while attempting to start a shell process", e, cmd_list
        ) from None

    # decoding the output and err and return as a string tuple
    return proc.returncode, out.decode("utf-8"), err.decode("utf-8")


def execute_async_cmd(cmd_list, cwd, env=None, out=PIPE, err=PIPE):
    """Execute an asynchronous command

    This function executes an asynchronous command and returns a
    popen subprocess object wrapped with psutil.

    :param cmd_list: list of command with arguments
    :type cmd_list: list of str
    :param cwd: current working directory
    :type cwd: str
    :param env: environment variables to set
    :type env: dict
    :return: the subprocess object
    :rtype: psutil.Popen
    """
    global verbose_shell
    if verbose_shell:
        logger.debug(f"Executing command: {' '.join(cmd_list)}")

    try:
        popen_obj = psutil.Popen(
            cmd_list, cwd=cwd, stdout=out, stderr=err, env=env, close_fds=True
        )
        time.sleep(0.2)
        popen_obj.poll()
        if not popen_obj.is_running() and popen_obj.returncode != 0:
            output, error = popen_obj.communicate()
            err_msg = ""
            if output:
                err_msg += output.decode("utf-8") + " "
            if error:
                err_msg += error.decode("utf-8")
            raise ShellError("Command failed immediately", err_msg, cmd_list)
    except OSError as e:
        raise ShellError("Failed to run command", e, cmd_list) from None
    return popen_obj
