import os.path as osp
import time

import pytest

from smartsim import Experiment, constants
from smartsim.database import LSFOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher != "lsf":
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_lsf_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    exp_name = "test-launch-lsf-orc-batch"
    exp = Experiment(exp_name, launcher="lsf")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = LSFOrchestrator(
        6780, batch=True, project=wlmutils.get_test_account(), time="00:05", smts=1
    )
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


def test_launch_lsf_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""
    exp_name = "test-launch-lsf-cluster-orc-batch"
    exp = Experiment(exp_name, launcher="lsf")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        project=wlmutils.get_test_account(),
        time="00:03",
        smts=1,
    )
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


def test_launch_lsf_cluster_orc_reconnect(fileutils, wlmutils):
    """test reconnecting to clustered 3-node orchestrator"""

    exp_name = "test-launch-lsf-cluster-orc-batch-reconect"
    exp = Experiment(exp_name, launcher="lsf")
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        project=wlmutils.get_test_account(),
        time="00:05",
        smts=1,
    )
    orc.set_path(test_dir)

    exp.start(orc, block=True)

    status = exp.get_status(orc)
    # don't use assert so that orc we don't leave an orphan process
    if constants.STATUS_FAILED in status:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    exp_name = "test-orc-lsf-cluster-orc-batch-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher="lsf")

    checkpoint = osp.join(test_dir, "smartsim_db.dat")
    reloaded_orc = exp_2.reconnect_orchestrator(checkpoint)

    # let statuses update once
    time.sleep(5)

    statuses = exp_2.get_status(reloaded_orc)
    for stat in statuses:
        if stat == constants.STATUS_FAILED:
            exp_2.stop(reloaded_orc)
            assert False
    exp_2.stop(reloaded_orc)


def test_orc_converter_lsf_batch():
    
    def converter(host):
        int_dict = {"host1": "HOST1-IB", "host2": "HOST2-IB"}
        if host in int_dict.keys():
            return int_dict[host]
        else:
            return ""

    orc = LSFOrchestrator(6780, db_nodes=3, batch=True, hosts=["batch", "host1", "host2"], hostname_converter=converter)
    assert orc.entities[0].hosts == ["HOST1-IB", "HOST2-IB"]
    assert orc.batch_settings.batch_args["m"]=="\"batch host1 host2\""
