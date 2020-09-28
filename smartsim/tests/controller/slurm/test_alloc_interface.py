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
def test_add_alloc():
    alloc = get_alloc_id()
    exp = Experiment("Test-Add-Allocation")
    exp.add_allocation(alloc)

    run_settings = {
        "ppn": 1,
        "nodes": 1,
        "executable": "python sleep.py",
        "alloc": alloc
    }
    model = exp.create_model("model", path=test_path, run_settings=run_settings)

    exp.start()
    exp.poll()
    status = exp.get_status(model)
    assert(status == "COMPLETED")

@controller_test
def test_get_release_allocation():
    """test getting and immediately releasing an allocation"""
    alloc_id = ctrl.get_allocation(nodes=1, ppn=1)
    ctrl.release(alloc_id=alloc_id)


# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id