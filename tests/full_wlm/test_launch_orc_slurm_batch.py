import time
import pytest
import os.path as osp
from smartsim import Experiment, constants
from smartsim.database import SlurmOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

def test_launch_slurm_orc(fileutils):
    """test single node orchestrator
    """
    exp_name = "test-launch-slurm-orc-batch"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = SlurmOrchestrator(6780, batch=True)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert(False)

    exp.stop(orc)
    status = exp.get_status(orc)
    assert(all([stat == constants.STATUS_CANCELLED for stat in status]))


def test_launch_slurm_cluster_orc(fileutils):
    """test clustered 3-node orchestrator
    """
    exp_name = "test-launch-slurm-cluster-orc-batch"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=True)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert(False)

    exp.stop(orc)
    status = exp.get_status(orc)
    assert(all([stat == constants.STATUS_CANCELLED for stat in status]))


def test_launch_slurm_cluster_orc_reconnect(fileutils):
    """test reconnecting to clustered 3-node orchestrator
    """

    exp_name = "test-launch-slurm-cluster-orc-batch-reconect"
    exp = Experiment(exp_name, launcher="slurm")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=True)
    orc.set_path(test_dir)

    exp.start(orc, block=True)

    status = exp.get_status(orc)
    # don't use assert so that orc we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert(False)

    exp_name = "test-orc-slurm-cluster-orc-batch-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher="slurm")

    checkpoint = osp.join(test_dir, "smartsim_db.dat")
    reloaded_orc = exp_2.reconnect_orchestrator(checkpoint)

    # let statuses update once
    time.sleep(5)

    statuses = exp_2.get_status(reloaded_orc)
    for stat in statuses:
        if stat == constants.STATUS_FAILED:
            exp_2.stop(reloaded_orc)
            assert(False)
    exp_2.stop(reloaded_orc)