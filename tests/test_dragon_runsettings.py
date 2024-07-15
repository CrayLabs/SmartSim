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

from smartsim.settings import DragonRunSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_a


def test_dragon_runsettings_nodes():
    """Verify that node count is set correctly"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    exp_value = 3
    rs.set_nodes(exp_value)
    assert rs.run_args["nodes"] == exp_value

    exp_value = 9
    rs.set_nodes(exp_value)
    assert rs.run_args["nodes"] == exp_value


def test_dragon_runsettings_tasks_per_node():
    """Verify that tasks per node is set correctly"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    exp_value = 3
    rs.set_tasks_per_node(exp_value)
    assert rs.run_args["tasks-per-node"] == exp_value

    exp_value = 7
    rs.set_tasks_per_node(exp_value)
    assert rs.run_args["tasks-per-node"] == exp_value


def test_dragon_runsettings_cpu_affinity():
    """Verify that the CPU affinity is set correctly"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    exp_value = [0, 1, 2, 3]
    rs.set_cpu_affinity([0, 1, 2, 3])
    assert rs.run_args["cpu-affinity"] == ",".join(str(val) for val in exp_value)

    # ensure the value is not changed when we extend the list
    exp_value.extend([4, 5, 6])
    assert rs.run_args["cpu-affinity"] != ",".join(str(val) for val in exp_value)

    rs.set_cpu_affinity(exp_value)
    assert rs.run_args["cpu-affinity"] == ",".join(str(val) for val in exp_value)

    # ensure the value is not changed when we extend the list
    rs.run_args["cpu-affinity"] = "7,8,9"
    assert rs.run_args["cpu-affinity"] != ",".join(str(val) for val in exp_value)


def test_dragon_runsettings_gpu_affinity():
    """Verify that the GPU affinity is set correctly"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    exp_value = [0, 1, 2, 3]
    rs.set_gpu_affinity([0, 1, 2, 3])
    assert rs.run_args["gpu-affinity"] == ",".join(str(val) for val in exp_value)

    # ensure the value is not changed when we extend the list
    exp_value.extend([4, 5, 6])
    assert rs.run_args["gpu-affinity"] != ",".join(str(val) for val in exp_value)

    rs.set_gpu_affinity(exp_value)
    assert rs.run_args["gpu-affinity"] == ",".join(str(val) for val in exp_value)

    # ensure the value is not changed when we extend the list
    rs.run_args["gpu-affinity"] = "7,8,9"
    assert rs.run_args["gpu-affinity"] != ",".join(str(val) for val in exp_value)


def test_dragon_runsettings_on_gpu():
    """Verify that setting gpu as a node feature works correctly and that
    using the function is not additive"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    feature_key = "node-feature"

    # check that no default features exist
    features = rs.run_args.get(feature_key, [])
    assert not features

    # set to initial value
    rs.set_node_feature("gpu")
    features = rs.run_args.get(feature_key, [])
    assert "gpu" in features

    # change value, verify overwrite
    rs.set_node_feature("cpu")
    features = rs.run_args.get(feature_key, [])
    assert "cpu" in features
    assert "gpu" not in features
