# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from ..util.shell import execute_cmd


def qstat(args):
    """Calls PBS qstat with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of qstat
    """
    cmd = ["qstat"] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def qsub(args):
    """Calls PBS qsub with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of salloc
    """
    cmd = ["qsub"] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def qdel(args):
    """Calls PBS qdel with args.

    returncode is also supplied in this function.

    :param args: list of command arguments
    :type args: list of str
    :return: output and error
    :rtype: str
    """
    cmd = ["qdel"] + args
    returncode, out, error = execute_cmd(cmd)
    return returncode, out, error
