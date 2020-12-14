import filecmp
from os import environ, getcwd, path
from shutil import which

import pytest

from smartsim import Experiment
from smartsim.utils.test.decorators import orchestrator_test_slurm

"""
NOTE: These tests will throw a warning in the second experiment
      because the orchestrator jobs are cancelled by a jobmanager
      that didn't launch them. This is normal and to be expected.
"""


# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(), "./orchestrator_test/")

# --- Tests  -----------------------------------------------
@pytest.mark.skip(reason="Requires client libraries to be installed")
@orchestrator_test_slurm
def test_db_cluster_reconnect_send_receive():
    """test that a database can be reconnected to and data can be
    retrieved

    First experiment populates the database and the second experiment
    uses the first experiment directory to communicate the needed
    information about the database location.
    """

    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp_1_dir = "/".join((test_path, "exp_1"))
    exp_2_dir = "/".join((test_path, "exp_2"))

    exp_1 = Experiment("exp_1")
    sim_dict = {
        "executable": "python",
        "nodes": 1,
        "alloc": db_test_alloc,
        "exe_args": "reconnect_sim.py --cluster",
    }
    O1 = exp_1.create_orchestrator(
        path=exp_1_dir, port=6780, db_nodes=3, dpn=3, alloc=db_test_alloc
    )
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start(O1, M1)
    exp_1.poll()
    assert "FAILED" not in exp_1.get_status(M1)

    # start the experiment that will reconnect to the database of the
    # first experiment.
    exp_2 = Experiment("exp_2")
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python",
        "alloc": db_test_alloc,
        "exe_args": " reconnect_node.py --cluster",
    }
    M2 = exp_2.create_model("M2", path=exp_2_dir, run_settings=node_settings)

    # before running, assert that the status of the database is "RUNNING"
    assert all(status == "RUNNING" for status in exp_2.get_status(O2))
    exp_2.start(M2, O2)
    exp_2.poll()
    exp_2.stop(O2)

    # compare the output of the sim and the node and ensure they are the same
    f1 = "/".join((exp_1_dir, "M1.out"))
    f2 = "/".join((exp_2_dir, "M2.out"))
    num_lines = sum(1 for line in open(f1))
    assert num_lines > 0
    assert filecmp.cmp(f1, f2)


@pytest.mark.skip(reason="Requires client libraries to be installed")
@orchestrator_test_slurm
def test_db_reconnect_send_receive():
    """test that a (non cluster) database can be reconnected to and data can be
    retrieved

    First experiment populates the database and the second experiment
    uses the first experiment directory to communicate the needed
    information about the database location.
    """

    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp_1_dir = "/".join((test_path, "exp_1"))
    exp_2_dir = "/".join((test_path, "exp_2"))

    exp_1 = Experiment("exp_1")
    sim_dict = {
        "executable": "python",
        "exe_args": "reconnect_sim.py",
        "nodes": 1,
        "alloc": db_test_alloc,
    }
    O1 = exp_1.create_orchestrator(
        path=exp_1_dir, port=6780, db_nodes=1, alloc=db_test_alloc
    )
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert "FAILED" not in exp_1.get_status(M1)

    # start the experiment that will reconnect to the database of the
    # first experiment.
    exp_2 = Experiment("exp_2")
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python",
        "exe_args": "reconnect_node.py",
        "alloc": db_test_alloc,
    }
    M2 = exp_2.create_model("M2", path=exp_2_dir, run_settings=node_settings)

    # before running, assert that the status of the database is "RUNNING"
    assert all(status == "RUNNING" for status in exp_2.get_status(O2))
    exp_2.start(M2, O2)
    exp_2.poll()
    exp_2.stop(O2)

    # compare the output of the sim and the node and ensure they are the same
    f1 = "/".join((exp_1_dir, "M1.out"))
    f2 = "/".join((exp_2_dir, "M2.out"))
    num_lines = sum(1 for line in open(f1))
    assert num_lines > 0
    assert filecmp.cmp(f1, f2)
