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
from smartsim._core.commands.launch_commands import LaunchCommands
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a

pre_cmd = Command(command=["pre", "cmd"])
launch_cmd = Command(command=["launch", "cmd"])
post_cmd = Command(command=["post", "cmd"])
pre_commands_list = CommandList(commands=[pre_cmd])
launch_command_list = CommandList(commands=[launch_cmd])
post_command_list = CommandList(commands=[post_cmd])


def test_launchCommand_init():
    launch_cmd = LaunchCommands(
        prelaunch_commands=pre_commands_list,
        launch_commands=launch_command_list,
        postlaunch_commands=post_command_list,
    )
    assert launch_cmd.prelaunch_command == pre_commands_list
    assert launch_cmd.launch_command == launch_command_list
    assert launch_cmd.postlaunch_command == post_command_list
