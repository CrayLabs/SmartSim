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

import pytest

from smartsim._core.commands.command import Command
from smartsim._core.commands.command_list import CommandList
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a

salloc_cmd = Command(command=["salloc", "-N", "1"])
srun_cmd = Command(command=["srun", "-n", "1"])
sacct_cmd = Command(command=["sacct", "--user"])


def test_command_init():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    assert cmd_list.commands == [salloc_cmd, srun_cmd]


def test_command_getitem_int():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    get_value = cmd_list[0]
    assert get_value == salloc_cmd


def test_command_getitem_slice():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    get_value = cmd_list[0:2]
    assert get_value == [salloc_cmd, srun_cmd]


def test_command_setitem_idx():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    cmd_list[0] = sacct_cmd
    for cmd in cmd_list.commands:
        assert cmd.command in [sacct_cmd.command, srun_cmd.command]


def test_command_setitem_slice():
    cmd_list = CommandList(commands=[srun_cmd, srun_cmd])
    cmd_list[0:2] = [sacct_cmd, sacct_cmd]
    for cmd in cmd_list.commands:
        assert cmd.command == sacct_cmd.command


def test_command_setitem_fail():
    cmd_list = CommandList(commands=[srun_cmd, srun_cmd])
    with pytest.raises(TypeError):
        cmd_list[0] = "fail"
    with pytest.raises(TypeError):
        cmd_list[0:1] = "fail"
    with pytest.raises(TypeError):
        cmd_list[0:1] = "fail"
    with pytest.raises(TypeError):
        _ = Command(command=["salloc", "-N", 1])
    with pytest.raises(TypeError):
        cmd_list[0:1] = [Command(command=["salloc", "-N", "1"]), Command(command=1)]


def test_command_delitem():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    del cmd_list.commands[0]
    assert cmd_list.commands == [srun_cmd]


def test_command_len():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    assert len(cmd_list) is 2


def test_command_insert():
    cmd_list = CommandList(commands=[salloc_cmd, srun_cmd])
    cmd_list.insert(0, sacct_cmd)
    assert cmd_list.commands == [sacct_cmd, salloc_cmd, srun_cmd]
