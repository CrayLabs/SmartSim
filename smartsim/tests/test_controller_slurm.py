import pytest
import time

from glob import glob
from decorator import decorator
from shutil import rmtree, which, copyfile
from os import getcwd, listdir, path, environ, mkdir, remove

from smartsim import Experiment
from smartsim.utils import get_logger
from smartsim.control import Controller
from smartsim.tests.decorators import slurm_controller_test
from smartsim.error import SmartSimError

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")
run_settings = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python sleep.py"
}

# experiment with non-clustered orchestrator
exp = Experiment("test")
M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
O1 = exp.create_orchestrator(path=test_path, cluster_size=1)
N1 = exp.create_node("n1",script_path=test_path, run_settings=run_settings)


# experiment with clustered orchestrator
exp_2 = Experiment("test_2")
C1 = exp_2.create_orchestrator(path=test_path, cluster_size=3)

# init slurm controller for testing
ctrl = Controller()
ctrl.init_launcher("slurm")

@slurm_controller_test
def test_ensemble():
    ctrl.start(ensembles=exp.ensembles)
    ctrl.poll(3, False, True)
    statuses = ctrl.get_ensemble_status(exp.ensembles[0])
    assert("FAILED" not in statuses)
    ctrl.release()

@slurm_controller_test
def test_orchestrator():
    ctrl.start(orchestrator=O1)
    time.sleep(5)
    statuses = ctrl.get_orchestrator_status(O1)
    assert("FAILED" not in statuses)

@slurm_controller_test
def test_cluster_orchestrator():
    ctrl.start(orchestrator=C1)
    time.sleep(10)
    statuses = ctrl.get_orchestrator_status(O1)
    assert("FAILED" not in statuses)

@slurm_controller_test
def test_node():
    ctrl.start(nodes=N1)
    while not ctrl.finished(N1):
        time.sleep(3)
    status = ctrl.get_node_status(N1)
    assert(status == "COMPLETED")
    ctrl.release()

@slurm_controller_test
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
    ctrl.release()

@slurm_controller_test
def test_stop_ensemble():
    ctrl.start(ensembles=exp.ensembles)
    time.sleep(5)
    ctrl.stop(ensembles=exp.ensembles)
    ctrl.release()

@slurm_controller_test
def test_stop_orchestrator():
    ctrl.start(orchestrator=O1)
    time.sleep(5)
    ctrl.stop(orchestrator=O1)
    ctrl.release()

@slurm_controller_test
def test_stop_all():
    ctrl.start(exp.ensembles, nodes=N1, orchestrator=O1)
    time.sleep(5)
    ctrl.stop(exp.ensembles, nodes=N1, orchestrator=O1)
    ctrl.release()


# Error handling test cases

run_settings_immediate_failure = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python bad.py"
}
run_settings_report_failure = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python bad.py",
    "exe_args": "--time 10"
}

exp_3 = Experiment("test_immediate_failure")
M3 = exp_3.create_model("m3", path=test_path, run_settings=run_settings_immediate_failure)

exp_4 = Experiment("test_report_failure")
M4 = exp_4.create_model("m4", path=test_path, run_settings=run_settings_report_failure)


@slurm_controller_test
def test_catch_failure():
    """Test when a failure inside a model occurs right after launch"""
    with pytest.raises(SmartSimError):
        ctrl.start(exp_3.ensembles)


@slurm_controller_test
def test_failed_status():
    """Test when a failure occurs deep into model execution"""
    ctrl.start(exp_4.ensembles)
    while not ctrl.finished(M4):
        time.sleep(3)
    status = ctrl.get_model_status(M4)
    assert(status == "FAILED")
    ctrl.release()
