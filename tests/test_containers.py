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


import os
from pathlib import Path
from shutil import which

import pytest

from smartsim import Experiment, status
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble
from smartsim.settings.containers import Singularity

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


# Check if singularity is available as command line tool
singularity_exists = which("singularity") is not None
containerURI = "docker://alrigazzi/smartsim-testing:latest"


def test_singularity_commands(fileutils):
    """Test generation of singularity commands."""

    # Note: We skip first element so singularity is not needed to run test

    c = Singularity(containerURI)
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec {containerURI}"

    c = Singularity(containerURI, args="--verbose")
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec --verbose {containerURI}"

    c = Singularity(containerURI, args=["--verbose", "--cleanenv"])
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec --verbose --cleanenv {containerURI}"

    c = Singularity(containerURI, mount="/usr/local/bin")
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec --bind /usr/local/bin {containerURI}"

    c = Singularity(containerURI, mount=["/usr/local/bin", "/lus/datasets"])
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec --bind /usr/local/bin,/lus/datasets {containerURI}"

    c = Singularity(
        containerURI,
        mount={
            "/usr/local/bin": "/bin",
            "/lus/datasets": "/datasets",
            "/cray/css/smartsim": None,
        },
    )
    cmd = " ".join(c._container_cmds()[1:])
    assert (
        cmd
        == f"exec --bind /usr/local/bin:/bin,/lus/datasets:/datasets,/cray/css/smartsim {containerURI}"
    )

    c = Singularity(containerURI, args="--verbose", mount="/usr/local/bin")
    cmd = " ".join(c._container_cmds()[1:])
    assert cmd == f"exec --verbose --bind /usr/local/bin {containerURI}"


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_basic(fileutils, test_dir):
    """Basic argument-less Singularity test"""

    container = Singularity(containerURI)

    exp = Experiment("singularity_basic", exp_path=test_dir, launcher="local")
    run_settings = exp.create_run_settings(
        "python3", "sleep.py --time=3", container=container
    )
    model = exp.create_model("singularity_basic", run_settings)

    script = fileutils.get_test_conf_path("sleep.py")
    model.attach_generator_files(to_copy=[script])
    exp.generate(model)

    exp.start(model, summary=False)

    # get and confirm status
    stat = exp.get_status(model)[0]
    assert stat == status.STATUS_COMPLETED

    print(exp.summary())


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_args(fileutils, test_dir):
    """Test combinations of args and mount arguments for Singularity"""

    hometest_dir = os.path.join(str(Path.home()), "test")  # $HOME/test
    mount_paths = {test_dir + "/singularity_args": hometest_dir}
    container = Singularity(containerURI, args="--contain", mount=mount_paths)

    exp = Experiment("singularity_args", launcher="local", exp_path=test_dir)

    run_settings = exp.create_run_settings(
        "python3", "test/check_dirs.py", container=container
    )
    model = exp.create_model("singularity_args", run_settings)
    script = fileutils.get_test_conf_path("check_dirs.py")
    model.attach_generator_files(to_copy=[script])
    exp.generate(model)

    exp.start(model, summary=False)

    # get and confirm status
    stat = exp.get_status(model)[0]
    assert stat == status.STATUS_COMPLETED

    print(exp.summary())


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_smartredis(test_dir, fileutils, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.

    Note: This is a containerized port of test_smartredis.py
    """

    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity(containerURI)

    rs = exp.create_run_settings(
        "python3", "producer.py --exchange", container=container
    )
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
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())
