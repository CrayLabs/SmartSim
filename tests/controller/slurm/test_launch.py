import time
import pytest
from shutil import which
from os import getcwd, path, environ

from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test


# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./controller_test/")
ctrl = Controller()

# --- Tests  -----------------------------------------------

# experiment with non-clustered orchestrator
exp = Experiment("test")

run_settings = {
    "ppn": 1,
    "nodes": 1,
    "executable": "python",
    "exe_args": "sleep.py"
}

@controller_test
def test_models():
    # setup allocation and models
    alloc = get_alloc_id()
    run_settings["alloc"] = alloc
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)

    ctrl.start(M1, M2)
    ctrl.poll(3, False, True)
    statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    assert("FAILED" not in statuses)

@controller_test
def test_orchestrator():
    # setup allocation and orchestrator
    alloc = get_alloc_id()
    O1 = exp.create_orchestrator(path=test_path, alloc=alloc)

    ctrl.start(O1)
    time.sleep(5)
    statuses = ctrl.get_entity_list_status(O1)
    assert("FAILED" not in statuses)
    ctrl.stop_entity_list(O1)


@controller_test
def test_ensemble():
    # setup allocation and orchestrator
    alloc = get_alloc_id()
    ensemble = exp.create_ensemble("e1")
    M3 = exp.create_model("m3", path=test_path, run_settings=run_settings)
    ensemble.add_model(M3)

    ctrl.start(ensemble)
    time.sleep(5)
    statuses = ctrl.get_entity_list_status(ensemble)
    assert("FAILED" not in statuses)
    ctrl.stop_entity_list(ensemble)

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id