import pytest
from smartsim import Experiment, constants
from smartsim.database import SlurmOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

def test_launch_slurm_orc(fileutils, wlmutils):
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


def test_launch_slurm_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator
    """

    # TODO detect number of nodes in allocation and skip if not sufficent
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

