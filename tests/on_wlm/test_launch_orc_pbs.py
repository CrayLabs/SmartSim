import pytest

from smartsim import Experiment, status
from smartsim.database import PBSOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_pbs_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    if launcher != "pbs":
        pytest.skip("Test only runs on systems with PBSPro as WLM")

    exp_name = "test-launch-pbs-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = PBSOrchestrator(6780, batch=False, interface=network_interface)
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


def test_launch_pbs_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator

    This test will fail if the PBS allocation is not
    obtained with `-l place=scatter`

    It will also fail if there are not enough nodes in the
    allocation to support a 3 node deployment
    """
    launcher = wlmutils.get_test_launcher()
    if launcher != "pbs":
        pytest.skip("Test only runs on systems with PBSPro as WLM")

    exp_name = "test-launch-pbs-cluster-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = PBSOrchestrator(
        6780, db_nodes=3, batch=False, inter_op_threads=4, interface=network_interface
    )
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    if len(orc.get_address()) < 3:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
