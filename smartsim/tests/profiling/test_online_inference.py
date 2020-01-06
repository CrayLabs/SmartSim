import os
import pytest

from smartsim import State, Controller

# Comment to run profiling tests
pytestmark = pytest.mark.skip()

def test_inference_path():
    """test the latency for a small amount of data for the inference
       path which means from client to node to client (roundtrip)
       Ave Inference Loop Time:  0.5269723296165466
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(20,)'" # literal_eval used
        num_packets = 20
        test_id = "_small_data"
        run_inference_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()


def test_inference_path_2D_huge():
    """
    Ave Inference Loop Time:  3.7058833241462708
    """
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(2000, 20000)'"
        num_packets = 20
        test_id = "_2D_huge"
        run_inference_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()


def test_inference_path_2D_1GB():
    """Ave Inference Loop Time:  10.20196316242218"""
    try:
        if os.environ["HOST"] != "cicero":
            pytest.skip()
        data_size = "'(4000, 30000)'"
        num_packets = 20
        test_id = "_2D_1GB"
        run_inference_path(data_size, num_packets, test_id)
    except KeyError:
        pytest.skip()

def run_inference_path(data_size, num_packets, test_id):
    state = State(experiment="online-training")
    train_settings = {
        "nodes": 1,
        "executable": "python node.py",
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets)
    }
    node_name = "training_node" + test_id
    sim_name = "sim-model" + test_id
    orc_name = "orchestrator" + test_id
    state.create_node(node_name, script_path=os.getcwd() + "/inference/", run_settings=train_settings)
    state.create_target("simulation", run_settings=sim_dict)
    state.create_model(sim_name, "simulation", path=os.getcwd() + "/inference/")
    state.create_orchestrator(orc_name)
    state.register_connection(sim_name, node_name)
    state.register_connection(node_name, sim_name)

    sim_control = Controller(state, launcher="slurm")
    sim_control.start()

def create_exe_args(data_size, num_packets):
    size = "--size=" + str(data_size)
    num_pack = "--num_packets=" + str(num_packets)
    return " ".join((size, num_pack))
