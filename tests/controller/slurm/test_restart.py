import time
from os import environ, getcwd, path
from shutil import which

import pytest

from smartsim import Experiment, constants
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(), "./controller_test/")
ctrl = Controller()

# --- Tests  -----------------------------------------------


@controller_test
def test_restart():

    # initialize experiment entities
    exp = Experiment("Restart-Test")
    alloc = get_alloc_id()
    run_settings = {
        "ntasks": 1,
        "nodes": 1,
        "executable": "python",
        "exe_args": "sleep.py",
        "alloc": alloc,
    }
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
    O1 = exp.create_orchestrator(path=test_path, port=6780, alloc=alloc)

    # start all entities for the first time
    ctrl.start(M1, M2, O1)
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert constants.STATUS_FAILED not in statuses

    ctrl.stop_entity(M1)  # should not change the status
    ctrl.stop_entity(M2)  # should not change the status
    ctrl.stop_entity_list(O1)  # should change status to cancelled
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    assert all(stat == constants.STATUS_COMPLETED for stat in model_statuses)
    assert all(stat == constants.STATUS_CANCELLED for stat in orc_status)

    # restart all entities
    ctrl.start(M1, M2, O1)
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert constants.STATUS_FAILED not in statuses

    # TODO: add job history check here
    ctrl.stop_entity(M1)
    ctrl.stop_entity(M2)
    ctrl.stop_entity_list(O1)
    model_statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = ctrl.get_entity_list_status(O1)
    assert all(stat == constants.STATUS_COMPLETED for stat in model_statuses)
    assert all(stat == constants.STATUS_CANCELLED for stat in orc_status)


# ------ Helper Functions ------------------------------------------


def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id
