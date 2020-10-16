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
def test_cluster_orchestrator():
    """Experiment with a cluster orchestrator"""

    exp = Experiment("Clustered-Orchestrator-Test")
    alloc = get_alloc_id()
    C1 = exp.create_orchestrator(path=test_path, db_nodes=3, alloc=alloc)

    ctrl.start(C1)
    time.sleep(10)
    statuses = ctrl.get_entity_list_status(C1)
    assert("FAILED" not in statuses)
    ctrl.stop_entity_list(C1)


# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id