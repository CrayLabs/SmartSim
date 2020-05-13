import pytest
import time

from glob import glob
from decorator import decorator
from shutil import rmtree, which, copyfile
from os import getcwd, listdir, path, environ, mkdir, remove

from smartsim import Experiment
from smartsim.utils import get_logger
from smartsim.control import Controller
from smartsim.tests.decorators import controller_test
from smartsim.error import SmartSimError, LauncherError

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")

# --- straightforward launching -------------------------------------

# experiment with non-clustered orchestrator
exp = Experiment("test")
ctrl = Controller()
alloc = ctrl.get_allocation(nodes=5, ppn=3)
run_settings = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python sleep.py",
    "alloc": alloc
}
M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
O1 = exp.create_orchestrator(path=test_path, alloc=alloc)
N1 = exp.create_node("n1",script_path=test_path, run_settings=run_settings)

@controller_test
def test_ensemble():
    ctrl.start(ensembles=exp.ensembles)
    ctrl.poll(3, False, True)
    statuses = ctrl.get_ensemble_status(exp.ensembles[0])
    assert("FAILED" not in statuses)

@controller_test
def test_orchestrator():
    ctrl.start(orchestrator=O1)
    time.sleep(5)
    statuses = ctrl.get_orchestrator_status(O1)
    assert("FAILED" not in statuses)
    ctrl.stop(orchestrator=O1)

@controller_test
def test_node():
    ctrl.start(nodes=N1)
    while not ctrl.finished(N1):
        time.sleep(3)
    status = ctrl.get_node_status(N1)
    assert(status == "COMPLETED")

@controller_test
def test_multiple_runs():
    """test calling start multiple times in a row"""
    ctrl.start(ensembles=exp.ensembles)
    ctrl.poll(3, False, True)
    statuses = ctrl.get_ensemble_status(exp.ensembles[0])
    assert("FAILED" not in statuses)

    ctrl.start(nodes=N1)
    ctrl.poll(3, False, True)
    statuses = ctrl.get_node_status(N1)
    assert("FAILED" not in statuses)

@controller_test
def test_all():
    ctrl.start(
        ensembles=exp.ensembles,
        nodes=N1,
        orchestrator=O1
    )
    ctrl.poll(3, False, True)
    ensemble_status = ctrl.get_ensemble_status(exp.ensembles[0])
    node_status = ctrl.get_node_status(N1)
    orc_status = ctrl.get_orchestrator_status(O1)
    statuses = orc_status + ensemble_status + [node_status]
    assert("FAILED" not in statuses)

# --- test stop -------------------------------------------------------

@controller_test
def test_stop_ensemble():
    ctrl.start(ensembles=exp.ensembles)
    time.sleep(3)
    ctrl.stop(ensembles=exp.ensembles)

@controller_test
def test_stop_orchestrator():
    ctrl.start(orchestrator=O1)
    time.sleep(3)
    ctrl.stop(orchestrator=O1)

@controller_test
def test_stop_all():
    ctrl.start(exp.ensembles, nodes=N1, orchestrator=O1)
    time.sleep(3)
    ctrl.stop(exp.ensembles, nodes=N1, orchestrator=O1)

@controller_test
def test_get_release_allocation():
    """test getting and immediately releasing an allocation"""
    alloc_id = ctrl.get_allocation(nodes=1, ppn=1)
    ctrl.release(alloc_id=alloc_id)


# ---- cluster orchestrator -------------------------------------------

# experiment with clustered orchestrator
exp_2 = Experiment("test_2")
C1 = exp_2.create_orchestrator_cluster(alloc, path=test_path, db_nodes=3)

@controller_test
def test_cluster_orchestrator():
    ctrl.start(orchestrator=C1)
    time.sleep(10)
    statuses = ctrl.get_orchestrator_status(C1)
    assert("FAILED" not in statuses)
    ctrl.stop(orchestrator=C1)

# --- multiple dpn ---------------------------------------------------

# test multiple orchestrator per node
exp_3 = Experiment("test_3")
O2 = exp_3.create_orchestrator_cluster(alloc, path=test_path, db_nodes=1, dpn=3)

@controller_test
def test_dpn():
    """test launching multiple databases per node"""
    ctrl.start(orchestrator=O2)
    time.sleep(5)
    statuses = ctrl.get_orchestrator_status(O2)
    assert("FAILED" not in statuses)
    ctrl.stop(orchestrator=O2)

# --- existing db files -----------------------------------------------

exp_3 = Experiment("test_3")
O3 = exp_3.create_orchestrator_cluster(alloc, path=test_path, db_nodes=3, dpn=3)

@controller_test
def test_db_file_removal():
    """test that existing .conf, .out, and .err do not prevent
       launch of database.
    """
    for dbnode in O3.dbnodes:
        for port in dbnode.ports:
            conf_file = "/".join((dbnode.path, dbnode._get_dbnode_conf_fname(port)))
            open(conf_file, 'w').close()
        out_file = dbnode.run_settings["out_file"]
        err_file = dbnode.run_settings["err_file"]
        open(err_file, 'w').close()
        open(out_file, 'w').close()
    ctrl.start(orchestrator=O3)
    time.sleep(5)
    statuses = ctrl.get_orchestrator_status(O3)
    assert("FAILED" not in statuses)
    ctrl.stop(orchestrator=O3)

# --- error handling ---------------------------------------------------

# Error handling test cases
run_settings_report_failure = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python bad.py",
    "exe_args": "--time 10",
    "alloc": alloc
}

run_settings_no_alloc = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python bad.py",
    "exe_args": "--time 10"
}

exp_4 = Experiment("test_report_failure")
M4 = exp_4.create_model("m4", path=test_path, run_settings=run_settings_report_failure)

exp_5 = Experiment("test_no_alloc")
M5 = exp_5.create_model("m5", path=test_path, run_settings=run_settings_no_alloc)

@controller_test
def test_failed_status():
    """Test when a failure occurs deep into model execution"""
    ctrl.start(exp_4.ensembles)
    while not ctrl.finished(M4):
        time.sleep(3)
    status = ctrl.get_model_status(M4)
    assert(status == "FAILED")

@controller_test
def test_start_no_allocs():
    """test when a user doesnt provide an allocation with entity run_settings"""
    with pytest.raises(SmartSimError):
        ctrl.start(exp_5.ensembles)

@controller_test
def test_bad_release():
    """Test when a non-existant alloc_id is given to release"""
    with pytest.raises(SmartSimError):
        ctrl.release(alloc_id=111111)


# --- clean-up --------------------------------------------------
# release the final allocation
def test_release():
    """Release the allocation used to run all these tests"""
    ctrl.release()
