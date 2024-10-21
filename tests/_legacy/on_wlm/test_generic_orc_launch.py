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
from smartsim.status import JobStatus

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_feature_store_auto(test_dir, wlmutils):
    """test single node feature store"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-feature_store"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        batch=False,
        interface=network_interface,
        single_cmd=False,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that we don't leave an orphan process
    if JobStatus.FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == JobStatus.CANCELLED for stat in statuses])


def test_launch_cluster_feature_store_single(test_dir, wlmutils):
    """test clustered 3-node feature store with single command"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-feature_store-single"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that feature_store we don't leave an orphan process
    if JobStatus.FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == JobStatus.CANCELLED for stat in statuses])


def test_launch_cluster_feature_store_multi(test_dir, wlmutils):
    """test clustered 3-node feature store with multiple commands"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-feature-store-multi"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store = exp.create_feature_store(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=False,
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.start(feature_store, block=True)
    statuses = exp.get_status(feature_store)

    # don't use assert so that feature_store we don't leave an orphan process
    if JobStatus.FAILED in statuses:
        exp.stop(feature_store)
        assert False

    exp.stop(feature_store)
    statuses = exp.get_status(feature_store)
    assert all([stat == JobStatus.CANCELLED for stat in statuses])
