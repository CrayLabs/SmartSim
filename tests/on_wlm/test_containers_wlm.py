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

from shutil import which

import pytest

from smartsim import Experiment
from smartsim.entity import Ensemble
from smartsim.settings.containers import Singularity
from smartsim.status import SmartSimStatus

"""Test SmartRedis container integration on a supercomputer with a WLM."""

# Check if singularity is available as command line tool
singularity_exists = which("singularity") is not None
containerURI = "docker://alrigazzi/smartsim-testing:latest"


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_wlm_smartredis(fileutils, test_dir, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.

    Note: This is a containerized port of test_smartredis.py for WLM system
    """

    launcher = wlmutils.get_test_launcher()
    print(launcher)
    if launcher not in ["pbs", "slurm"]:
        pytest.skip(
            f"Test only runs on systems with PBS or Slurm as WLM. Current launcher: {launcher}"
        )

    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher=launcher
    )

    # create and start a database
    orc = exp.create_database(
        port=wlmutils.get_test_port(), interface=wlmutils.get_test_interface()
    )
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity(containerURI)
    rs = exp.create_run_settings(
        "python3", "producer.py --exchange", container=container
    )
    rs.set_tasks(1)
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer",
        params=params,
        run_settings=rs,
        perm_strat="step",
    )

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    if not all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())
