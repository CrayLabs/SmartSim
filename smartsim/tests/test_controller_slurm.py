import pytest
import time

from glob import glob
from decorator import decorator
from shutil import rmtree, which, copyfile
from os import getcwd, listdir, path, environ, mkdir, remove

from smartsim import Experiment
from smartsim.control import Controller
from smartsim.tests.decorators import slurm_controller_test

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
    ctrl.release()

@slurm_controller_test
def test_orchestrator():
    ctrl.start(orchestrator=O1)
    ctrl.poll(3, False, True)
    ctrl.release()

@slurm_controller_test
def test_cluster_orchestrator():
    ctrl.start(orchestrator=O1)
    ctrl.poll(3, False, True)
    ctrl.release()

@slurm_controller_test
def test_node():
    ctrl.start(nodes=N1)
    ctrl.poll(3, False, True)
    ctrl.release()

@slurm_controller_test
def test_all():
    ctrl.start(
        ensembles=exp.ensembles,
        nodes=N1,
        orchestrator=O1
    )
    ctrl.poll(3, False, True)
    ctrl.release()

@slurm_controller_test
def test_stop_ensemble():
    ctrl.start(ensembles=exp.ensembles)
    time.sleep(10)
    ctrl.stop(ensembles=exp.ensembles)
    ctrl.release()

@slurm_controller_test
def test_stop_orchestrator():
    ctrl.start(orchestrator=O1)
    time.sleep(10)
    ctrl.stop(stop_orchestrator=True)
    ctrl.release()

@slurm_controller_test
def test_stop_all():
    ctrl.start(exp.ensembles, nodes=N1, orchestrator=O1)
    time.sleep(10)
    ctrl.stop(exp.ensembles, nodes=N1, stop_orchestrator=True)
    ctrl.release()