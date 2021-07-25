import time

import pytest

from smartsim import Experiment, constants
from smartsim.database import LSFOrchestrator
from smartsim.error import SSUnsupportedError

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_lsf_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    if launcher != "lsf":
        pytest.skip("Test only runs on systems with LSF as WLM")

    exp_name = "test-launch-lsf-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = LSFOrchestrator(6780, batch=False)
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

    time.sleep(5)


def test_launch_lsf_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""

    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()
    if launcher != "lsf":
        pytest.skip("Test only runs on systems with LSF as WLM")

    exp_name = "test-launch-lsf-cluster-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = LSFOrchestrator(6780, db_nodes=3, batch=False)
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

    time.sleep(5)


def test_orc_converter_lsf(wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher != "lsf":
        pytest.skip("Test only runs on systems with LSF as WLM")

    def converter(host):
        int_dict = {"host1": "HOST1-IB", "host2": "HOST2-IB"}
        if host in int_dict.keys():
            return int_dict[host]
        else:
            return ""

    orc = LSFOrchestrator(6780, db_nodes=3, batch=False, hosts=["batch", "host1", "host2"], hostname_converter=converter)
    assert orc.entities[0].hosts == ["HOST1-IB", "HOST2-IB"]

    orc = LSFOrchestrator(6780, db_nodes=3, batch=False, hosts=["batch", "host1", "host2"], hostname_converter=None)
    assert orc.entities[0].hosts == ["batch", "host1", "host2"]


def test_catch_orc_errors_lsf(wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher != "lsf":
        pytest.skip("Test only runs on systems with LSF as WLM")

    with pytest.raises(SSUnsupportedError):
        orc = LSFOrchestrator(6780, db_nodes=2, db_per_host=2, batch=False, hosts=["host1", "host2"])