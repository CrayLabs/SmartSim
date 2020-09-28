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


@controller_test
def test_dpn():
    """test launching multiple databases per node"""
    exp = Experiment("Multi-DPN-Test")
    alloc = get_alloc_id()
    Orc = exp.create_orchestrator(path=test_path, db_nodes=1, dpn=3, alloc=alloc)

    ctrl.start(orchestrator=Orc)
    time.sleep(5)
    statuses = ctrl.get_orchestrator_status(Orc)
    assert("FAILED" not in statuses)
    ctrl.stop(orchestrator=Orc)


# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id