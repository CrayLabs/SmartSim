import os
import zmq
import pickle
import psutil
from ..error import ShellError, LauncherError, SSConfigError
from subprocess import PIPE, Popen, CalledProcessError, TimeoutExpired, run


from ..utils import get_logger, get_env

logger = get_logger(__name__)

try:
    level = get_env("SMARTSIM_LOG_LEVEL")
    verbose_shell = True if level == "developer" else False
except SSConfigError:
    verbose_shell = False


def ping_host(hostname):
    """Ping a hostname and get the IPv4 address of the node.

    :param hostname: hostname of the node
    :type hostname: str
    :raises ShellError: if ping to host fails
    :return: output of ping command
    :rtype: str
    """
    try:
        returncode, out, err = execute_cmd(["ping -c 1 " + hostname], shell=True)
        return out
    except LauncherError as e:
        raise ShellError(f"Ping to {hostname} failed.") from e


def execute_cmd(
    cmd_list, shell=False, cwd=None, env=None, proc_input="", timeout=None
):
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
        logger.debug("Executing %s cmd: %s" % (source, " ".join(cmd_list)))

    # spawning the subprocess and connecting to its output
    proc = Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=cwd, shell=shell, env=env
    )
    try:
        proc_input = proc_input.encode("utf-8")
        out, err = proc.communicate(input=proc_input, timeout=timeout)
    except TimeoutExpired as e:
        proc.kill()
        output, errs = proc.communicate()
        raise ShellError("Failed to execute command, timeout reached", e,  cmd_list)
    except OSError as e:
        raise ShellError("Exception while attempting to start a shell process",
                         e, cmd_list)

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
        logger.debug("Executing async Popen cmd: %s" % " ".join(cmd_list))

    try:
        popen_obj = psutil.Popen(
            cmd_list, cwd=cwd, stdout=out, stderr=err, env=env, close_fds=True
        )
    except OSError as err:
        raise ShellError("Failed to run async shell command", err, cmd_list)
    return popen_obj
