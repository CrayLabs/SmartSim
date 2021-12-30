import pytest

from smartsim import Experiment, status
from smartsim.database import CobaltOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_cobalt_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    if launcher != "cobalt":
        pytest.skip("Test only runs on systems with Cobalt as WLM")

    exp_name = "test-launch-cobalt-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = CobaltOrchestrator(6780, batch=False, interface=network_interface)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in status:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    status = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in status])


def test_launch_cobalt_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator

    It will also fail if there are not enough nodes in the
    allocation to support a 3 node deployment
    """
    launcher = wlmutils.get_test_launcher()
    if launcher != "cobalt":
        pytest.skip("Test only runs on systems with Cobalt as WLM")

    exp_name = "test-launch-cobalt-cluster-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = CobaltOrchestrator(
        6780, db_nodes=3, batch=False, inter_op_threads=4, interface=network_interface
    )
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    status = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in status:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    status = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in status])
