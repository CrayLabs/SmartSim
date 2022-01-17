import os.path as osp
import time

import pytest

from smartsim import Experiment, status
from smartsim.database import SlurmOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_slurm_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    exp_name = "test-launch-slurm-orc-batch"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, batch=True, interface=network_interface)
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


def test_launch_slurm_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""
    exp_name = "test-launch-slurm-cluster-orc-batch"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=True, interface=network_interface)
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


def test_launch_slurm_cluster_orc_reconnect(fileutils, wlmutils):
    """test reconnecting to clustered 3-node orchestrator"""

    exp_name = "test-launch-slurm-cluster-orc-batch-reconect"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=True, interface=network_interface)
    orc.set_path(test_dir)

    exp.start(orc, block=True)

    statuses = exp.get_status(orc)
    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    exp_name = "test-orc-slurm-cluster-orc-batch-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher="slurm")

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
