import os.path as osp
import time

from smartsim import Experiment, status
from smartsim.database import Orchestrator

first_dir = ""

# TODO ensure database is shutdown
# use https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test


def test_local_orchestrator(fileutils):
    """Test launching orchestrator locally"""
    global first_dir
    exp_name = "test-orc-launch-local"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)
    first_dir = test_dir

    orc = Orchestrator(port=6780)
    orc.set_path(test_dir)

    exp.start(orc)
    statuses = exp.get_status(orc)
    assert [stat != status.STATUS_FAILED for stat in statuses]

    # simulate user shutting down main thread
    exp._control._jobs.actively_monitoring = False
    exp._control._launcher.task_manager.actively_monitoring = False


def test_reconnect_local_orc():
    """Test reconnecting to orchestrator from first experiment"""
    global first_dir
    # start new experiment
    exp_name = "test-orc-local-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher="local")

    checkpoint = osp.join(first_dir, "smartsim_db.dat")
    reloaded_orc = exp_2.reconnect_orchestrator(checkpoint)

    # let statuses update once
    time.sleep(5)

    statuses = exp_2.get_status(reloaded_orc)
    for stat in statuses:
        if stat == status.STATUS_FAILED:
            exp_2.stop(reloaded_orc)
            assert False
    exp_2.stop(reloaded_orc)
