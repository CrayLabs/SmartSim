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
from smartsim.status import SmartSimStatus

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_orc_auto(test_dir, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-orc"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        batch=False,
        interface=network_interface,
        single_cmd=False,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_single(test_dir, wlmutils):
    """test clustered 3-node orchestrator with single command"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-single"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_multi(test_dir, wlmutils):
    """test clustered 3-node orchestrator with multiple commands"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-multi"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=False,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])
