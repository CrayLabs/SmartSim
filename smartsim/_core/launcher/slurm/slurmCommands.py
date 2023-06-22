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

import typing as t

from ....error import LauncherError
from ...utils.helpers import expand_exe_path
from ..util.shell import execute_cmd


def sstat(args: t.List[str]) -> t.Tuple[str, str]:
    """Calls sstat with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sstat
    """
    _sstat = _find_slurm_command("sstat")
    cmd = [_sstat] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def sacct(args: t.List[str]) -> t.Tuple[str, str]:
    """Calls sacct with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sacct
    """
    _sacct = _find_slurm_command("sacct")
    cmd = [_sacct] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def salloc(args: t.List[str]) -> t.Tuple[str, str]:
    """Calls slurm salloc with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of salloc
    """
    _salloc = _find_slurm_command("salloc")
    cmd = [_salloc] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def sinfo(args: t.List[str]) -> t.Tuple[str, str]:
    """Calls slurm sinfo with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sinfo
    """
    _sinfo = _find_slurm_command("sinfo")
    cmd = [_sinfo] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def scontrol(args: t.List[str]) -> t.Tuple[str, str]:
    """Calls slurm scontrol with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sinfo
    """
    _scontrol = _find_slurm_command("scontrol")
    cmd = [_scontrol] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def scancel(args: t.List[str]) -> t.Tuple[int, str, str]:
    """Calls slurm scancel with args.

    returncode is also supplied in this function.

    :param args: list of command arguments
    :type args: list of str
    :return: output and error
    :rtype: str
    """
    _scancel = _find_slurm_command("scancel")
    cmd = [_scancel] + args
    returncode, out, error = execute_cmd(cmd)
    return returncode, out, error


def _find_slurm_command(cmd: str) -> str:
    try:
        full_cmd = expand_exe_path(cmd)
        return full_cmd
    except (TypeError, FileNotFoundError) as e:
        raise LauncherError(
            f"Slurm Launcher could not find path of {cmd} command"
        ) from e
