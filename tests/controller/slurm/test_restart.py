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

@controller_test
def test_restart():

    # initialize experiment entities
    exp = Experiment("Restart-Test")
    alloc = get_alloc_id()
    run_settings = {
        "ppn": 1,
        "nodes": 1,
        "executable": "python",
        "exe_args": "sleep.py",
        "alloc": alloc
    }
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
    O1 = exp.create_orchestrator(path=test_path, alloc=alloc)

    # start all entities for the first time
    ctrl.start(M1, M2, O1)
    ctrl.poll(3, False, True)
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert("FAILED" not in statuses)

    ctrl.stop_entity(M1)
    ctrl.stop_entity(M2)
    ctrl.stop_entity_list(O1)


    # restart all entities
    ctrl.start(M1, M2, O1)
    ctrl.poll(3, False, True)
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert("FAILED" not in statuses)

    # TODO: add job history check here
    ctrl.stop_entity(M1)
    ctrl.stop_entity(M2)
    ctrl.stop_entity_list(O1)

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id