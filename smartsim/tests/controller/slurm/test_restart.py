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
def test_restart():

    # initialize experiment entities
    exp = Experiment("Restart-Test")
    alloc = get_alloc_id()
    run_settings = {
        "ppn": 1,
        "nodes": 1,
        "executable": "python sleep.py",
        "alloc": alloc
    }
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
    O1 = exp.create_orchestrator(path=test_path, alloc=alloc)
    N1 = exp.create_node("n1", path=test_path, run_settings=run_settings)

    # start all entities for the first time
    ctrl.start(
        ensembles=exp.ensembles,
        nodes=N1,
        orchestrator=O1
    )
    ctrl.poll(3, False, True)
    ensemble_status = ctrl.get_ensemble_status(exp.ensembles[0])
    node_status = ctrl.get_node_status(N1)
    orc_status = ctrl.get_orchestrator_status(O1)
    statuses = orc_status + ensemble_status + [node_status]
    assert("FAILED" not in statuses)

    ctrl.stop(ensembles=exp.ensembles, nodes=N1, orchestrator=O1)

    # restart all entities
    ctrl.start(
        ensembles=exp.ensembles,
        nodes=N1,
        orchestrator=O1
    )
    ctrl.poll(3, False, True)
    ensemble_status = ctrl.get_ensemble_status(exp.ensembles[0])
    node_status = ctrl.get_node_status(N1)
    orc_status = ctrl.get_orchestrator_status(O1)
    statuses = orc_status + ensemble_status + [node_status]
    assert("FAILED" not in statuses)

    # TODO: add job history query here
    ctrl.stop(ensembles=exp.ensembles, nodes=N1, orchestrator=O1)

# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id