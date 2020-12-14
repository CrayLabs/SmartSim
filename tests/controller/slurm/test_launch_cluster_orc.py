import time
from os import environ, getcwd, path

from smartsim import Experiment, constants
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
    C1 = exp.create_orchestrator(path=test_path, port=6780, db_nodes=3, alloc=alloc)

    ctrl.start(C1, block=False)
    time.sleep(10)
    statuses = ctrl.get_entity_list_status(C1)
    assert(all([stat == constants.STATUS_RUNNING for stat in statuses]))
    ctrl.stop_entity_list(C1)
    statuses = ctrl.get_entity_list_status(C1)
    assert(all([stat == constants.STATUS_CANCELLED for stat in statuses]))

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id