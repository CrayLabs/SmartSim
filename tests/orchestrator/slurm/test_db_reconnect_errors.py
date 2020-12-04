import pytest
from os import path, getcwd, environ

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.utils.test.decorators import orchestrator_test_slurm

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./orchestrator_test/")

# --- Tests  -----------------------------------------------

@pytest.mark.skip(reason="Requires client libraries to be installed")
@orchestrator_test_slurm
def test_db_reconnection_not_running_failure():

    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("Reconnect-DB-Exp")
    sim_dict = {
        "executable": "python",
        "exe_args": "reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, port=6780, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start(O1, M1)
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))
    exp_1.stop(O1)
    exp_1.poll(poll_db=True)

    exp_2 = Experiment("Reconnect-DB-exp-error-checker")
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)

@orchestrator_test_slurm
def test_db_reconnection_no_file():

    exp = Experiment("DB-error-no-reconnect-file")
    with pytest.raises(SmartSimError):
        O2 = exp.reconnect_orchestrator(getcwd())
