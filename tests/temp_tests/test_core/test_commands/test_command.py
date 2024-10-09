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

pytestmark = pytest.mark.group_a


def test_command_init():
    cmd = Command(command=["salloc", "-N", "1"])
    assert cmd.command == ["salloc", "-N", "1"]


def test_command_invalid_init():
    cmd = Command(command=["salloc", "-N", "1"])
    assert cmd.command == ["salloc", "-N", "1"]


def test_command_getitem_int():
    with pytest.raises(TypeError):
        _ = Command(command=[1])
    with pytest.raises(TypeError):
        _ = Command(command=[])


def test_command_getitem_slice():
    cmd = Command(command=["salloc", "-N", "1"])
    get_value = cmd[0:2]
    assert get_value.command == ["salloc", "-N"]


def test_command_setitem_int():
    cmd = Command(command=["salloc", "-N", "1"])
    cmd[0] = "srun"
    cmd[1] = "-n"
    assert cmd.command == ["srun", "-n", "1"]


def test_command_setitem_slice():
    cmd = Command(command=["salloc", "-N", "1"])
    cmd[0:2] = ["srun", "-n"]
    assert cmd.command == ["srun", "-n", "1"]


def test_command_setitem_fail():
    cmd = Command(command=["salloc", "-N", "1"])
    with pytest.raises(TypeError):
        cmd[0] = 1
    with pytest.raises(TypeError):
        cmd[0:2] = [1, "-n"]


def test_command_delitem():
    cmd = Command(
        command=["salloc", "-N", "1", "--constraint", "P100"],
    )
    del cmd.command[3]
    del cmd.command[3]
    assert cmd.command == ["salloc", "-N", "1"]


def test_command_len():
    cmd = Command(command=["salloc", "-N", "1"])
    assert len(cmd) is 3


def test_command_insert():
    cmd = Command(command=["-N", "1"])
    cmd.insert(0, "salloc")
    assert cmd.command == ["salloc", "-N", "1"]
