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

from copy import deepcopy

import pytest

from smartsim import Experiment
from smartsim._core.utils.helpers import is_valid_cmd
from smartsim.status import SmartSimStatus

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_mpmd(fileutils, test_dir, wlmutils):
    """Run an MPMD model twice

    and check that it always gets executed the same way.
    All MPMD-compatible run commands which do not
    require MPI are tested.

    This test requires three nodes to run.
    """
    exp_name = "test-mpmd"
    launcher = wlmutils.get_test_launcher()
    # MPMD is supported in LSF, but the test for it is different
    mpmd_supported = ["slurm", "pbs"]
    if launcher not in mpmd_supported:
        pytest.skip("Test requires Slurm, or PBS to run")

    # aprun returns an error if the launched app is not an MPI exec
    # as we do not want to add mpi4py as a dependency, we prefer to
    # skip this test for aprun
    by_launcher = {
        "slurm": ["srun", "mpirun"],
        "pbs": ["mpirun"],
    }

    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    def prune_commands(launcher):
        available_commands = []
        if launcher in by_launcher:
            for cmd in by_launcher[launcher]:
                if is_valid_cmd(cmd):
                    available_commands.append(cmd)
        return available_commands

    run_commands = prune_commands(launcher)
    if len(run_commands) == 0:
        pytest.skip(
            f"MPMD on {launcher} only supported for run commands {by_launcher[launcher]}"
        )

    for run_command in run_commands:
        script = fileutils.get_test_conf_path("sleep.py")
        settings = exp.create_run_settings(
            "python", f"{script} --time=5", run_command=run_command
        )
        settings.set_tasks(1)

        settings.make_mpmd(deepcopy(settings))
        settings.make_mpmd(deepcopy(settings))

        mpmd_model = exp.create_model(
            f"mpmd-{run_command}", path=test_dir, run_settings=settings
        )
        exp.start(mpmd_model, block=True)
        statuses = exp.get_status(mpmd_model)
        assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])

        exp.start(mpmd_model, block=True)
        statuses = exp.get_status(mpmd_model)
        assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])
