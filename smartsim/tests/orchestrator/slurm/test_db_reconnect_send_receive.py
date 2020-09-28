import pytest
import filecmp
from shutil import which
from os import path, getcwd, environ

from smartsim import Experiment
from smartsim.tests.decorators import orchestrator_test_slurm

"""
NOTE: These tests will throw a warning in the second experiment
      because the orchestrator jobs are cancelled by a jobmanager
      that didn't launch them. This is normal and to be expected.
"""


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
def test_db_cluster_reconnect_send_receive():
    """test that a database can be reconnected to and data can be
    retrieved

    First experiment populates the database and the second experiment
    uses the first experiment directory to communicate the needed
    information about the database location.
    """

    global db_test_alloc
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    exp_1.add_allocation(db_test_alloc)
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc,
        "exe_args":"--cluster"
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir,
                                   db_nodes=3, dpn=3, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    # start the experiment that will reconnect to the database of the
    # first experiment.
    exp_2 = Experiment("exp_2")
    exp_2.add_allocation(db_test_alloc)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python reconnect_node.py",
        "alloc": db_test_alloc,
        "exe_args":"--cluster"
    }
    N2 = exp_2.create_node("N2", path=exp_2_dir, run_settings=node_settings)

    # before running, assert that the status of the database is "RUNNING"
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.start(ssnodes=[N2], orchestrator=O2)
    exp_2.poll()
    exp_2.stop(orchestrator=O2)

    # compare the output of the sim and the node and ensure they are the same
    f1 = "/".join((exp_1_dir,"M1.out"))
    f2 = "/".join((exp_2_dir,"N2.out"))
    num_lines = sum(1 for line in open(f1))
    assert(num_lines>0)
    assert(filecmp.cmp(f1,f2))


@orchestrator_test_slurm
def test_db_reconnect_send_receive():
    """test that a (non cluster) database can be reconnected to and data can be
    retrieved

    First experiment populates the database and the second experiment
    uses the first experiment directory to communicate the needed
    information about the database location.
    """

    global db_test_alloc
    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    exp_1.add_allocation(db_test_alloc)
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir,
                                   db_nodes=1, alloc=db_test_alloc)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    # start the experiment that will reconnect to the database of the
    # first experiment.
    exp_2 = Experiment("exp_2")
    exp_2.add_allocation(db_test_alloc)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python reconnect_node.py",
        "alloc": db_test_alloc
    }
    N2 = exp_2.create_node("N2", path=exp_2_dir, run_settings=node_settings)

    # before running, assert that the status of the database is "RUNNING"
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.start(ssnodes=[N2], orchestrator=O2)
    exp_2.poll()
    exp_2.stop(orchestrator=O2)

    # compare the output of the sim and the node and ensure they are the same
    f1 = "/".join((exp_1_dir,"M1.out"))
    f2 = "/".join((exp_2_dir,"N2.out"))
    num_lines = sum(1 for line in open(f1))
    assert(num_lines>0)
    assert(filecmp.cmp(f1,f2))