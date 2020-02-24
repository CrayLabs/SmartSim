import os
import pytest
import time

from smartsim import State, Controller, Generator
from distutils import dir_util
from shutil import which, copyfile

# Comment to run profiling tests
#pytestmark = pytest.mark.skip()

# control wether the test runs with a database cluster or not
CLUSTER=True

def test_train_path():
    """test the latency for a sending a vector, a matrix, and a 3D tensor"""

    # test data sizes, and ID
    # literal eval is used to create sizes
    data = ["'(200,)'", "'(200,200)'", "'(200, 200, 200)'"]
    test_ids = ["_1D", "_2D", "_3D"]

    # see if we are on slurm machine
    if not which("srun"):
        pytest.skip()
    cluster_size = 1
    if CLUSTER:
        cluster_size = 3
    for data_size, test_id in zip(data, test_ids):
        num_packets = 20
        run_train_path(data_size, num_packets, test_id, cluster_size)

def run_train_path(data_size, num_packets, test_id, cluster_size):
    experiment_dir = "".join(("online-training", test_id, "/"))
    state = State(experiment=experiment_dir)
    node_name = "training_node" + test_id
    sim_name = "sim-model" + test_id

    # Setup training loop with one model, one cluster db, and
    # on node for training
    train_settings = {
        "nodes": 1,
        "executable": "python node.py",
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets)
    }
    state.create_node(node_name,
                      run_settings=train_settings)
    state.create_model(sim_name,
                       run_settings=sim_dict)
    state.create_orchestrator(cluster_size=cluster_size)
    state.register_connection(sim_name, node_name)

    # generate experiment directory
    generator = Generator(state,
                          model_files="./training/simulation.py")
    generator.generate()

    # TODO generator should copy files over
    # for now, copy over node script
    copyfile(os.getcwd() + "/training/node.py",
             experiment_dir + "node.py")

    control = Controller(state, launcher="slurm")
    control.start()
    while not control.finished():
        time.sleep(3)
    control.release()


def create_exe_args(data_size, num_packets):
    size = "--size=" + str(data_size)
    num_pack = "--num_packets=" + str(num_packets)
    return " ".join((size, num_pack))
