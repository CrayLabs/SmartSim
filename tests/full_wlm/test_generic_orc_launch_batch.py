import pytest
import time

import os.path as osp
from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_orc_auto_batch(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-orc-batch"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        6780, batch=True, interface=network_interface, single_cmd=False
    )
    if wlmutils.get_test_launcher() == "lsf":
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_walltime("00:05")
    if wlmutils.get_test_launcher() == "cobalt":
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_queue("debug-flat-quad")
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])


def test_launch_cluster_orc_batch_single(fileutils, wlmutils):
    """test clustered 3-node orchestrator with single command"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-batch-single"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        6780, db_nodes=3, batch=True, interface=network_interface, single_cmd=True
    )
    if wlmutils.get_test_launcher() == "lsf":
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_walltime("00:05")
    if wlmutils.get_test_launcher() == "cobalt":
        # As Cobalt won't allow us to run two
        # jobs in the same debug queue, we need
        # to make sure the previous test's one is over
        time.sleep(30)
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_queue("debug-flat-quad")
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


def test_launch_cluster_orc_batch_multi(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""
    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()

    exp_name = "test-launch-auto-cluster-orc-batch-multi"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        6780, db_nodes=3, batch=True, interface=network_interface, single_cmd=False
    )
    if wlmutils.get_test_launcher() == "lsf":
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_walltime("00:05")
    if wlmutils.get_test_launcher() == "cobalt":
        # As Cobalt won't allow us to run two
        # jobs in the same debug queue, we need
        # to make sure the previous test's one is over
        time.sleep(30)
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_queue("debug-flat-quad")
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

    
def test_launch_cluster_orc_reconnect(fileutils, wlmutils):
    """test reconnecting to clustered 3-node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    exp_name = "test-launch-cluster-orc-batch-reconect"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(6780, db_nodes=3, batch=True, interface=network_interface)
    orc.set_path(test_dir)
    if wlmutils.get_test_launcher() == "lsf":
        orc.batch_settings.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        # As Cobalt won't allow us to run two
        # jobs in the same debug queue, we need
        # to make sure the previous test's one is over
        time.sleep(30)
        orc.batch_settings.set_account(wlmutils.get_test_account())
        orc.batch_settings.set_queue("debug-flat-quad")

    exp.start(orc, block=True)

    statuses = exp.get_status(orc)
    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    exp_name = "test-orc-cluster-orc-batch-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher=launcher)

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
