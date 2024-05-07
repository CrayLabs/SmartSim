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
import pathlib
import time

import pytest

from smartsim import Experiment
from smartsim.status import SmartSimStatus

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

if (pytest.test_launcher == "pbs") and (not pytest.has_aprun):
    pytestmark = pytest.mark.skip(
        reason="Launching feature stores in a batch job is not supported on PBS without ALPS"
    )


def test_launch_feature_store_auto_batch(test_dir, wlmutils):
    """test single node feature store"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-feature-store-batch"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        batch=True,
        interface=network_interface,
        single_cmd=False,
    )

    feature_store.batch_settings.set_account(wlmutils.get_test_account())

    feature_store.batch_settings.set_walltime("00:02:00")

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_feature_store_batch_single(test_dir, wlmutils):
    """test clustered 3-node feature store with single command"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-feature-store-batch-single"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        interface=network_interface,
        single_cmd=True,
    )

    feature_store.batch_settings.set_account(wlmutils.get_test_account())

    feature_store.batch_settings.set_walltime("00:02:00")

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that feature_store we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_feature_store_batch_multi(test_dir, wlmutils):
    """test clustered 3-node feature store"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-feature-store-batch-multi"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        interface=network_interface,
        single_cmd=False,
    )

    feature_store.batch_settings.set_account(wlmutils.get_test_account())

    feature_store.batch_settings.set_walltime("00:03:00")

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that feature_store we don't leave an orphan process
    if SmartSimStatus.STATUS_FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == SmartSimStatus.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_feature_store_reconnect(test_dir, wlmutils):
    """test reconnecting to clustered 3-node feature store"""
    p_test_dir = pathlib.Path(test_dir)
    launcher = wlmutils.get_test_launcher()
    exp_name = "test-launch-cluster-feature-store-batch-reconect"
    exp_1_dir = p_test_dir / exp_name
    exp_1_dir.mkdir()
    exp = Experiment(exp_name, launcher=launcher, exp_path=str(exp_1_dir))

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(), fs_nodes=3, batch=True, interface=network_interface
    )

    feature_store.batch_settings.set_account(wlmutils.get_test_account())

    feature_store.batch_settings.set_walltime("00:03:00")

    exp.start(feature_store, block=True)

    statuses = exp.get_status(feature_store)
    try:
        assert all(stat == SmartSimStatus.STATUS_RUNNING for stat in statuses)
    except Exception:
        exp.stop(feature_store)
        raise

    exp_name = "test-feature_store-cluster-feature-store-batch-reconnect-2nd"
    exp_2_dir = p_test_dir / exp_name
    exp_2_dir.mkdir()
    exp_2 = Experiment(exp_name, launcher=launcher, exp_path=str(exp_2_dir))

    try:
        checkpoint = osp.join(feature_store.path, "smartsim_db.dat")
        reloaded_feature_store = exp_2.reconnect_feature_store(checkpoint)

        # let statuses update once
        time.sleep(5)

        statuses = exp_2.get_status(reloaded_feature_store)
        assert all(stat == SmartSimStatus.STATUS_RUNNING for stat in statuses)
    except Exception:
        # Something went wrong! Let the experiment that started the DB
        # clean up the DB
        exp.stop(feature_store)
        raise

    try:
        # Test experiment 2 can stop the DB
        exp_2.stop(reloaded_feature_store)
        assert all(
            stat == SmartSimStatus.STATUS_CANCELLED
            for stat in exp_2.get_status(reloaded_feature_store)
        )
    except Exception:
        # Something went wrong! Let the experiment that started the DB
        # clean up the DB
        exp.stop(feature_store)
        raise
    else:
        # Ensure  it is the same DB that Experiment 1 was tracking
        time.sleep(5)
        assert not any(
            stat == SmartSimStatus.STATUS_RUNNING for stat in exp.get_status(feature_store)
        )
