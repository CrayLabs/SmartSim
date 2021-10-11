import pytest

from smartsim import Experiment, constants
from smartsim.database import SlurmOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_slurm_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")

    exp_name = "test-launch-slurm-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, batch=False, interface=network_interface)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    status = exp.get_status(orc)
    assert all([stat == constants.STATUS_CANCELLED for stat in status])


def test_launch_slurm_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""

    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")

    exp_name = "test-launch-slurm-cluster-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=False, interface=network_interface)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    status = exp.get_status(orc)
    assert all([stat == constants.STATUS_CANCELLED for stat in status])
