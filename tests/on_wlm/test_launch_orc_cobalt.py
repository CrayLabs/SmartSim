import pytest

from smartsim import Experiment, constants
from smartsim.database import CobaltOrchestrator
from smartsim.error import SmartSimError

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
    orc = CobaltOrchestrator(6780, batch=False)
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
    orc = CobaltOrchestrator(6780, db_nodes=3, batch=False, inter_op_threads=4)
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


def test_set_run_arg():
    orc = CobaltOrchestrator(6780, db_nodes=3, batch=False)
    orc.set_run_arg("account", "ACCOUNT")
    assert all([db.run_settings.run_args["account"]=="ACCOUNT" for db in orc.entities])
    orc.set_run_arg("pes-per-numa-node", "2")
    assert all(["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities])
    orc.set_cpus(4)
    assert all([db.run_settings.run_args["cpus-per-pe"] == 4 for db in orc.entities])


def test_set_batch_arg():
    orc = CobaltOrchestrator(6780, db_nodes=3, batch=False)
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = CobaltOrchestrator(6780, db_nodes=3, batch=True)
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    orc2.set_batch_arg("outputprefix", "new_output/")
    assert "outputprefix" not in orc2.batch_settings.batch_args