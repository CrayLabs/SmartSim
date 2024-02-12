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

import os.path as osp
import time

import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

if (pytest.test_launcher == "pbs") and (not pytest.has_aprun):
    pytestmark = pytest.mark.skip(
        reason="Launching orchestrators in a batch job is not supported on PBS without ALPS"
    )


def test_launch_orc_auto_batch(test_dir, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-orc-batch"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        batch=True,
        interface=network_interface,
        single_cmd=False,
    )

    orc.batch_settings.set_account(wlmutils.get_test_account())

    orc.batch_settings.set_walltime("00:02:00")
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_batch_single(test_dir, wlmutils):
    """test clustered 3-node orchestrator with single command"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-batch-single"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface=network_interface,
        single_cmd=True,
    )

    orc.batch_settings.set_account(wlmutils.get_test_account())

    orc.batch_settings.set_walltime("00:02:00")
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_batch_multi(test_dir, wlmutils):
    """test clustered 3-node orchestrator"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-batch-multi"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface=network_interface,
        single_cmd=False,
    )

    orc.batch_settings.set_account(wlmutils.get_test_account())

    orc.batch_settings.set_walltime("00:03:00")
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_reconnect(test_dir, wlmutils):
    """test reconnecting to clustered 3-node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    exp_name = "test-launch-cluster-orc-batch-reconect"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(), db_nodes=3, batch=True, interface=network_interface
    )
    orc.set_path(test_dir)

    orc.batch_settings.set_account(wlmutils.get_test_account())

    orc.batch_settings.set_walltime("00:03:00")

    exp.start(orc, block=True)

    statuses = exp.get_status(orc)
    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    exp_name = "test-orc-cluster-orc-batch-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher=launcher)

    checkpoint = osp.join(test_dir, "smartsim_db.dat")
    reloaded_orc = exp_2.reconnect_orchestrator(checkpoint)

    # let statuses update once
    time.sleep(5)

    statuses = exp_2.get_status(reloaded_orc)
    for stat in statuses:
        if stat == status.STATUS_FAILED:
            exp_2.stop(reloaded_orc)
            assert False

    exp_2.stop(reloaded_orc)
