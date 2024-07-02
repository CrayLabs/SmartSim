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

from ....error import LauncherError
from ....log import get_logger
from ...utils.helpers import expand_exe_path
from ...utils.shell import execute_cmd

logger = get_logger(__name__)


def sstat(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[str, str]:
    """Calls sstat with args

    :param args: List of command arguments
    :returns: Output and error of sstat
    """
    _, out, err = _execute_slurm_cmd("sstat", args, raise_on_err=raise_on_err)
    return out, err


def sacct(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[str, str]:
    """Calls sacct with args

    :param args: List of command arguments
    :returns: Output and error of sacct
    """
    _, out, err = _execute_slurm_cmd("sacct", args, raise_on_err=raise_on_err)
    return out, err


def salloc(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[str, str]:
    """Calls slurm salloc with args

    :param args: List of command arguments
    :returns: Output and error of salloc
    """
    _, out, err = _execute_slurm_cmd("salloc", args, raise_on_err=raise_on_err)
    return out, err


def sinfo(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[str, str]:
    """Calls slurm sinfo with args

    :param args: List of command arguments
    :returns: Output and error of sinfo
    """
    _, out, err = _execute_slurm_cmd("sinfo", args, raise_on_err=raise_on_err)
    return out, err


def scontrol(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[str, str]:
    """Calls slurm scontrol with args

    :param args: List of command arguments
    :returns: Output and error of sinfo
    """
    _, out, err = _execute_slurm_cmd("scontrol", args, raise_on_err=raise_on_err)
    return out, err


def scancel(args: t.List[str], *, raise_on_err: bool = False) -> t.Tuple[int, str, str]:
    """Calls slurm scancel with args.

    returncode is also supplied in this function.

    :param args: list of command arguments
    :return: output and error
    """
    return _execute_slurm_cmd("scancel", args, raise_on_err=raise_on_err)


def _find_slurm_command(cmd: str) -> str:
    try:
        full_cmd = expand_exe_path(cmd)
        return full_cmd
    except (TypeError, FileNotFoundError) as e:
        raise LauncherError(
            f"Slurm Launcher could not find path of {cmd} command"
        ) from e


def _execute_slurm_cmd(
    command: str, args: t.List[str], raise_on_err: bool = False
) -> t.Tuple[int, str, str]:
    cmd_exe = _find_slurm_command(command)
    cmd = [cmd_exe] + args
    returncode, out, error = execute_cmd(cmd)
    if returncode != 0:
        msg = f"An error occurred while calling {command}: {error}"
        if raise_on_err:
            raise LauncherError(msg)
        logger.error(msg)
    return returncode, out, error
