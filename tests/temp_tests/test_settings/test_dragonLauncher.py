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

from smartsim._core.launcher.dragon.dragon_launcher import (
    _as_run_request_args_and_policy,
)
from smartsim._core.schemas.dragon_requests import DragonRunPolicy, DragonRunRequestView
from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.dragon import DragonLaunchArguments
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Dragon)
    assert ls.launch_args.launcher_str() == LauncherType.Dragon.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nodes", id="set_nodes"),
        pytest.param(
            "set_tasks_per_node", (2,), "2", "tasks_per_node", id="set_tasks_per_node"
        ),
    ],
)
def test_dragon_class_methods(function, value, flag, result):
    dragonLauncher = LaunchSettings(launcher=LauncherType.Dragon)
    assert isinstance(dragonLauncher._arguments, DragonLaunchArguments)
    getattr(dragonLauncher.launch_args, function)(*value)
    assert dragonLauncher.launch_args._launch_args[flag] == result


NOT_SET = object()


@pytest.mark.parametrize("nodes", (NOT_SET, 20, 40))
@pytest.mark.parametrize("tasks_per_node", (NOT_SET, 1, 20))
@pytest.mark.parametrize("cpu_affinity", (NOT_SET, [1], [1, 2, 3]))
@pytest.mark.parametrize("gpu_affinity", (NOT_SET, [1], [1, 2, 3]))
def test_formatting_launch_args_into_request(
    nodes, tasks_per_node, cpu_affinity, gpu_affinity, test_dir
):
    launch_args = DragonLaunchArguments({})
    if nodes is not NOT_SET:
        launch_args.set_nodes(nodes)
    if tasks_per_node is not NOT_SET:
        launch_args.set_tasks_per_node(tasks_per_node)
    if cpu_affinity is not NOT_SET:
        launch_args.set_cpu_affinity(cpu_affinity)
    if gpu_affinity is not NOT_SET:
        launch_args.set_gpu_affinity(gpu_affinity)
    req, policy = _as_run_request_args_and_policy(
        launch_args, ("echo", "hello", "world"), test_dir, {}, "output.txt", "error.txt"
    )

    expected_args = {
        k: v
        for k, v in {
            "nodes": nodes,
            "tasks_per_node": tasks_per_node,
        }.items()
        if v is not NOT_SET
    }
    expected_run_req = DragonRunRequestView(
        exe="echo",
        exe_args=["hello", "world"],
        path=test_dir,
        env={},
        output_file="output.txt",
        error_file="error.txt",
        **expected_args,
    )
    assert req.exe == expected_run_req.exe
    assert req.exe_args == expected_run_req.exe_args
    assert req.nodes == expected_run_req.nodes
    assert req.tasks_per_node == expected_run_req.tasks_per_node
    assert req.hostlist == expected_run_req.hostlist
    assert req.pmi_enabled == expected_run_req.pmi_enabled
    assert req.path == expected_run_req.path
    assert req.output_file == expected_run_req.output_file
    assert req.error_file == expected_run_req.error_file

    expected_run_policy_args = {
        k: v
        for k, v in {"cpu_affinity": cpu_affinity, "gpu_affinity": gpu_affinity}.items()
        if v is not NOT_SET
    }
    assert policy == DragonRunPolicy(**expected_run_policy_args)
