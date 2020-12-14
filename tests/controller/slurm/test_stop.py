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

# experiment with non-clustered orchestrator
exp = Experiment("Stop-Tests")

run_settings = {"ntasks": 1, "nodes": 1, "executable": "python", "exe_args": "sleep.py"}


@controller_test
def test_stop_entity():
    # setup allocation and models
    alloc = get_alloc_id()
    run_settings["alloc"] = alloc
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)

    ctrl.start(M1, block=False)
    time.sleep(3)
    ctrl.stop_entity(M1)
    assert M1.name in ctrl._jobs.completed
    assert ctrl.get_entity_status(M1) == constants.STATUS_CANCELLED


@controller_test
def test_stop_entity_list():
    # setup allocation and orchestrator
    alloc = get_alloc_id()
    O1 = exp.create_orchestrator(path=test_path, alloc=alloc, port=6780)

    ctrl.start(O1, block=False)
    time.sleep(5)
    ctrl.stop_entity_list(O1)
    statuses = ctrl.get_entity_list_status(O1)
    assert all([orc.name in ctrl._jobs.completed for orc in O1.entities])
    assert all([stat == constants.STATUS_CANCELLED for stat in statuses])


# ------ Helper Functions ------------------------------------------


def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id
