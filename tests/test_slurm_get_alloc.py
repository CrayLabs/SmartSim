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


from smartsim.slurm import _get_alloc_cmd


def test_get_alloc_format():
    time = "10:00:00"
    nodes = 5
    account = "A35311"
    options = {"ntasks-per-node": 5}
    alloc_cmd = _get_alloc_cmd(nodes, time, account, options)
    result = [
        "--no-shell",
        "-N",
        "5",
        "-J",
        "SmartSim",
        "-t",
        "10:00:00",
        "-A",
        "A35311",
        "--ntasks-per-node=5",
    ]
    assert alloc_cmd == result


def test_get_alloc_format_overlap():
    """Test get alloc with collision between arguments and options"""
    time = "10:00:00"
    nodes = 5
    account = "A35311"
    options = {"N": 10, "time": "15", "account": "S1242"}
    alloc_cmd = _get_alloc_cmd(nodes, time, account, options)
    result = [
        "--no-shell",
        "-N",
        "5",
        "-J",
        "SmartSim",
        "-t",
        "10:00:00",
        "-A",
        "A35311",
    ]
    assert result == alloc_cmd
