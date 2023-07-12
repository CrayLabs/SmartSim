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

import psutil
import time
import typing as t
from subprocess import PIPE, TimeoutExpired

from ....error import ShellError
from ....log import get_logger
from ...utils.helpers import check_dev_log_level

logger = get_logger(__name__)
VERBOSE_SHELL = check_dev_log_level()


def execute_cmd(
    cmd_list: t.List[str],
    shell: bool = False,
    cwd: t.Optional[str] = None,
    env: t.Optional[t.Dict[str, str]] = None,
    proc_input: str = "",
    timeout: t.Optional[int] = None,
) -> t.Tuple[int, str, str]:
    """Execute a command locally

    :param cmd_list: list of command with arguments
    :type cmd_list: list of str
    :param shell: run in system shell, defaults to False
    :type shell: bool, optional
    :param cwd: current working directory, defaults to None
    :type cwd: str, optional
    :param env: environment to launcher process with,
                defaults to None (current env)
    :type env: dict[str, str], optional
    :param proc_input: input to the process, defaults to ""
    :type proc_input: str, optional
    :param timeout: timeout of the process, defaults to None
    :type timeout: int, optional
    :raises ShellError: if timeout of process was exceeded
    :raises ShellError: if child process raises an error
    :return: returncode, output, and error of the process
    :rtype: tuple of (int, str, str)
    """
    if VERBOSE_SHELL:
        source = "shell" if shell else "Popen"
        logger.debug(f"Executing {source} cmd: {' '.join(cmd_list)}")

    # spawning the subprocess and connecting to its output
    proc = psutil.Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=cwd, shell=shell, env=env
    )
    try:
        proc_bytes = proc_input.encode("utf-8")
        out, err = proc.communicate(input=proc_bytes, timeout=timeout)
    except TimeoutExpired as e:
        proc.kill()
        _, errs = proc.communicate()
        logger.error(errs)
        raise ShellError(
            "Failed to execute command, timeout reached", cmd_list, details=e
        ) from None
    except OSError as e:
        raise ShellError(
            "Exception while attempting to start a shell process", cmd_list, details=e
        ) from None

    # decoding the output and err and return as a string tuple
    return proc.returncode, out.decode("utf-8"), err.decode("utf-8")


def execute_async_cmd(
    cmd_list: t.List[str],
    cwd: str,
    env: t.Optional[t.Dict[str, str]] = None,
    out: int = PIPE,
    err: int = PIPE,
) -> psutil.Popen:
    """Execute an asynchronous command

    This function executes an asynchronous command and returns a
    popen subprocess object wrapped with psutil.

    :param cmd_list: list of command with arguments
    :type cmd_list: list of str
    :param cwd: current working directory
    :type cwd: str
    :param env: environment variables to set
    :type env: dict[str, str]
    :return: the subprocess object
    :rtype: psutil.Popen
    """
    if VERBOSE_SHELL:
        logger.debug(f"Executing command: {' '.join(cmd_list)}")

    try:
        popen_obj = psutil.Popen(
            cmd_list, cwd=cwd, stdout=out, stderr=err, env=env, close_fds=True
        )
        time.sleep(0.2)
        popen_obj.poll()
    except OSError as e:
        raise ShellError("Failed to run command", cmd_list, details=e) from None

    err_msg = ""
    try:
        if not popen_obj.is_running() and popen_obj.returncode != 0:
            output, error = popen_obj.communicate()
            if output:
                err_msg += output.decode("utf-8") + " "
            if error:
                err_msg += error.decode("utf-8")
            raise ShellError("Command failed immediately", cmd_list, details=err_msg)
    except OSError as e:
        raise ShellError("Failed to run command", cmd_list, details=e) from None

    return popen_obj
