import time
from os import getcwd, path, environ

from smartsim import constants
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./controller_test/")
ctrl = Controller()

# --- Tests  -----------------------------------------------


@controller_test
def test_dpn():
    """test launching multiple databases per node"""
    exp = Experiment("Multi-DPN-Test")
    alloc = get_alloc_id()
    Orc = exp.create_orchestrator(path=test_path, port=6780, db_nodes=1, dpn=3, alloc=alloc)

    ctrl.start(Orc, block=False)
    time.sleep(15)
    statuses = ctrl.get_entity_list_status(Orc)
    assert(all([stat == constants.STATUS_RUNNING for stat in statuses]))
    ctrl.stop_entity_list(Orc)
    statuses = ctrl.get_entity_list_status(Orc)
    assert(all([stat == constants.STATUS_CANCELLED for stat in statuses]))

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id