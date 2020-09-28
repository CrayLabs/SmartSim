import time
import pytest
from shutil import which
from os import getcwd, path, environ

from smartsim import Experiment
from smartsim.control import Controller
from smartsim.tests.decorators import controller_test


# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./controller_test/")

ctrl = Controller()

def test_setup_alloc():
    """Not a test, just used to ensure that at test time, the
       allocation is added to the controller. This has to be a
       test because otherwise it will run on pytest startup.
    """
    if not which("srun"):
        pytest.skip()
    alloc_id = environ["TEST_ALLOCATION_ID"]
    ctrl.add_allocation(alloc_id)
    assert("TEST_ALLOCATION_ID" in environ)

# --- Tests  -----------------------------------------------

# experiment with non-clustered orchestrator
exp = Experiment("Stop-Tests")

run_settings = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python sleep.py"
}

@controller_test
def test_stop_ensemble():
    # setup allocation and models
    alloc = get_alloc_id()
    run_settings["alloc"] = alloc
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)

    ctrl.start(ensembles=exp.ensembles)
    time.sleep(3)
    ctrl.stop(ensembles=exp.ensembles)

@controller_test
def test_stop_orchestrator():
    # setup allocation and orchestrator
    alloc = get_alloc_id()
    O1 = exp.create_orchestrator(path=test_path, alloc=alloc)

    ctrl.start(orchestrator=O1)
    time.sleep(5)
    ctrl.stop(orchestrator=O1)

@controller_test
def test_stop_node():
    # setup allocation and nodes
    alloc = get_alloc_id()
    run_settings["alloc"] = alloc
    N1 = exp.create_node("n1", path=test_path, run_settings=run_settings)

    ctrl.start(nodes=N1)
    time.sleep(3)
    ctrl.stop(nodes=N1)

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id