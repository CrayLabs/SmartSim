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

from smartsim import Experiment
from smartsim.settings import SrunSettings

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Test is only for Slurm WLM systems")


def test_mpmd_errors(monkeypatch):
    monkeypatch.setenv("SLURM_HET_SIZE", "1")
    exp_name = "test-het-job-errors"
    exp = Experiment(exp_name, launcher="slurm")
    rs: SrunSettings = exp.create_run_settings("sleep", "1", run_command="srun")
    rs2: SrunSettings = exp.create_run_settings("sleep", "1", run_command="srun")
    with pytest.raises(ValueError):
        rs.make_mpmd(rs2)

    monkeypatch.delenv("SLURM_HET_SIZE")
    rs.make_mpmd(rs2)
    with pytest.raises(ValueError):
        rs.set_het_group(1)


def test_set_het_groups(monkeypatch):
    """Test ability to set one or more het groups to run setting"""
    monkeypatch.setenv("SLURM_HET_SIZE", "4")
    exp_name = "test-set-het-group"
    exp = Experiment(exp_name, launcher="slurm")
    rs: SrunSettings = exp.create_run_settings("sleep", "1", run_command="srun")
    rs.set_het_group([1])
    assert rs.run_args["het-group"] == "1"
    rs.set_het_group([3, 2])
    assert rs.run_args["het-group"] == "3,2"
    with pytest.raises(ValueError):
        rs.set_het_group([4])


def test_orch_single_cmd(monkeypatch, wlmutils):
    """Test that single cmd is rejected in a heterogeneous job"""
    monkeypatch.setenv("SLURM_HET_SIZE", "1")
    exp_name = "test-orch-single-cmd"
    exp = Experiment(exp_name, launcher="slurm")
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=wlmutils.get_test_interface(),
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
    )

    for node in orc:
        assert node.is_mpmd == False
