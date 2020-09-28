import os
import zmq
import pickle
from ..error import ShellError, LauncherError, SSConfigError
from subprocess import PIPE, Popen, CalledProcessError, TimeoutExpired, run
from ..remote import CmdClient


from ..utils import get_logger, get_env
logger = get_logger(__name__)

try:
    level = get_env("SMARTSIM_LOG_LEVEL")
    verbose_shell = True if level == "developer" else False
except SSConfigError:
    verbose_shell = False


def is_remote():
    """Determine if a command should be sent to a CmdServer

    This function determines if a command should be sent to
    a CmdSerer running elsewhere by examining SMARTSIM_REMOTE env
    var.

    :return: true if command should be a RemoteRequest
    :rtype: bool
    """
    if "SMARTSIM_REMOTE" in os.environ:
            return True
    return False

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


def execute_cmd(cmd_list, shell=False, cwd=None, env=None, proc_input="",
                timeout=None, remote=True):
    """Execute a command locally or remotely

    This function executes a command either locally or remotely depending
    on the configuration set by the user for this experiment. If SMARSIM_REMOTE
    is set, send the command over the network to a CmdServer listening
    on a tcp socket. Otherwise execute the command locally.

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
    :param remote: disable remote sending(for CmdServer),
                   defaults to True
    :type remote: bool, optional
    :raises ShellError: if timeout of process was exceeded
    :raises ShellError: if child process raises an error
    :return: returncode, output, and error of the process
    :rtype: tuple of (int, str, str)
    """
    global verbose_shell

    # run the command remotely using CmdClient
    if is_remote() and remote:
        client = CmdClient()
        request = client.create_remote_request(
            cmd_list, shell=shell, cwd=cwd, proc_input=proc_input,
            env=env, timeout=timeout)
        return client.execute_remote_request(request)

    if verbose_shell:
        source = "shell" if shell else "Popen"
        logger.debug("Executing %s cmd: %s" % (source, " ".join(cmd_list)))

    # spawning the subprocess and connecting to its output
    proc = Popen(cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE,
                 cwd=cwd, shell=shell, env=env)
    try:
        proc_input = proc_input.encode("utf-8")
        out, err = proc.communicate(input=proc_input, timeout=timeout)
    except TimeoutExpired as e:
        proc.kill()
        output, errs = proc.communicate()
        logger.error("Timeout for command execution exceeded")
        raise ShellError("Failed to execute command: " + " ".join(cmd_list))
    except OSError as e:
        logger.error("Exception while attempting to start a shell process")
        raise ShellError("Failed to execute command: " + " ".join(cmd_list))

    # decoding the output and err and return as a string tuple
    return proc.returncode, out.decode('utf-8'), err.decode('utf-8')


def execute_async_cmd(cmd_list, cwd, remote=True):
    """Execute an asynchronous command

    This function executes an asynchronous command either
    locally or through zmq to a CmdServer listening over TCP.

    :param cmd_list: list of command with arguments
    :type cmd_list: list of str
    :param cwd: current working directory
    :type cwd: str
    :param remote: disable remote(for CmdServer), defaults to True
    :type remote: bool, optional
    :return: returncode and placeholders for output and error
    :rtype: tuple of (int, str, str)
    """
    global verbose_shell
    if verbose_shell:
        logger.debug("Executing async Popen cmd: %s" % " ".join(cmd_list))

    if is_remote() and remote:
        client = CmdClient()
        request = client.create_remote_request(cmd_list, cwd=cwd, is_async=True)
        return client.execute_remote_request(request)

    try:
        # TODO change to if remote than pipe the output
        popen_obj = Popen(cmd_list, cwd=cwd, stdout=PIPE, stderr=PIPE)
    except OSError as err:
        return popen_obj, -1, "", ""
    return popen_obj, 1, "", ""
