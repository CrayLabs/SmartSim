import os
import pytest

from smartsim import State, Controller


def test_train_path_small_data():
    """test the latency for a small amount of data
       in this case we are sending 20 vectors of length 20
       Average Train Path latency: 0.6775458931922913
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(20,)'" # literal_eval used
        num_packets = 20
        test_id = "_small_data"
        run_train_path(data_size, num_packets, test_id)
    # not on a linux machine or Cray
    except KeyError:
        pytest.skip()


def test_train_path_med_data():
    """test the latency for a small amount of data
       in this case we are sending 20 vectors of length 200000
       Average Train Path latency: 0.766495943069458
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(2000000,)'"
        num_packets = 20
        test_id = "_med_data"
        run_train_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()

def test_train_path_2D_small():
    """test the latency for a small amount of 2D data
       in this case we are sending 20 matricies of shape (200, 200)
       Average Train Path latency: 0.6400160551071167
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(200, 200)'"
        num_packets = 20
        test_id = "_2D_small"
        run_train_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()

def test_train_path_2D_huge():
    """test the latency for a small amount of 2D data
       in this case we are sending 20 matricies of shape (2000, 200000)
       Average Train Path latency: 8.110368180274964
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(2000, 20000)'"
        num_packets = 20
        test_id = "_2D_huge"
        run_train_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()


def test_train_path_2D_1GB():
    """test the latency for a small amount of 2D data of size 1gb
       in this case we are sending 20 matricies of shape (4000, 30000)
       Average Train Path latency: 23.108077692985535
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(4000, 30000)'"
        num_packets = 20
        test_id = "_2D_1GB"
        run_train_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()


def run_train_path(data_size, num_packets, test_id):
    state = State(experiment="online-training")
    train_settings = {
        "launcher": "slurm",
        "nodes": 1,
        "executable": "node.py",
        "run_command": "srun python"
    }
    node_name = "training_node" + test_id
    sim_name = "sim-model" + test_id
    orc_name = "orchestrator" + test_id
    state.create_node(node_name, script_path=os.getcwd() + "/training/", **train_settings)
    state.create_target("simulation")
    state.create_model(sim_name, "simulation", path=os.getcwd() + "/training/")
    state.create_orchestrator(orc_name)
    state.register_connection(sim_name, node_name)

    sim_dict = {
        "launcher": "slurm",
        "executable": "simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets),
        "run_command": "srun python"
    }
    sim_control = Controller(state, **sim_dict)
    sim_control.start()

def create_exe_args(data_size, num_packets):
    size = "--size=" + str(data_size)
    num_pack = "--num_packets=" + str(num_packets)
    return " ".join((size, num_pack))
