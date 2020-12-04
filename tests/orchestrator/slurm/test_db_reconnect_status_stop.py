import pytest
import filecmp
from shutil import which
from os import path, getcwd, environ

from smartsim import Experiment
from smartsim.utils.test.decorators import orchestrator_test_slurm

"""
NOTE: These tests will throw a warning in the second experiment
      because the orchestrator jobs are cancelled by a jobmanager
      that didn't launch them. This is normal and to be expected.
"""

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./orchestrator_test/")

# --- Tests  -----------------------------------------------
@pytest.mark.skip(reason="Requires client libraries to be installed")
@orchestrator_test_slurm
def test_db_cluster_reconnect_status_and_stop():
    """Tests status and stop with cluster and non-clustered
       orchestrators that have been reconnected.
    """

    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    sim_dict = {
        "executable": "python",
        "nodes": 1,
        "alloc": db_test_alloc,
        "exe_args":"reconnect_sim.py --cluster"
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, port=6780, db_nodes=3, dpn=2, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)

    # start first experiment
    exp_1.start(O1, M1)
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    # start second experiment with reconnected orchestrator
    exp_2 = Experiment("exp_2")
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))

    # cleanup and assert
    exp_2.stop(O2)
    exp_2.poll(poll_db=True)
    assert("RUNNING" not in exp_2.get_status(O2))

@pytest.mark.skip(reason="Requires client libraries to be installed")
@orchestrator_test_slurm
def test_db_reconnect_status_and_stop():

    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    sim_dict = {
        "executable": "python",
        "exe_args":  "reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, port=6780, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)

    # start first experiment
    exp_1.start(O1, M1)
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    # start second experiment with orchestrator from first experiment
    exp_2 = Experiment("exp_2")
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))

    # cleanup and check
    exp_2.stop(O2)
    exp_2.poll(poll_db=True)
    assert("RUNNING" not in exp_2.get_status(O2))
