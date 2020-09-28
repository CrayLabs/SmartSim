import pytest
import filecmp
from shutil import which
from os import path, getcwd, environ

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.tests.decorators import orchestrator_test_slurm

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./orchestrator_test/")
db_test_alloc = None

def test_setup_alloc():
    """Not a test, just used to ensure that at test time, the
       allocation is added to the controller. This has to be a
       test because otherwise it will run on pytest startup.
    """
    global db_test_alloc
    if not which("srun"):
        pytest.skip()
    assert("TEST_ALLOCATION_ID" in environ)
    db_test_alloc = environ["TEST_ALLOCATION_ID"]

# --- Tests  -----------------------------------------------

@orchestrator_test_slurm
def test_db_reconnection_not_running_failure():

    global db_test_alloc
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("Reconnect-DB-Exp")
    exp_1.add_allocation(db_test_alloc)

    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))
    exp_1.stop(orchestrator=O1)
    exp_1.poll(poll_db=True)

    exp_2 = Experiment("Reconnect-DB-exp-error-checker")
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)

@orchestrator_test_slurm
def test_db_reconnection_no_file():

    exp = Experiment("DB-error-no-reconnect-file")
    with pytest.raises(SmartSimError):
        O2 = exp.reconnect_orchestrator(getcwd())
