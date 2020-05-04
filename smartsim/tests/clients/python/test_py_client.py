import os
import pytest
import time

from smartsim import Experiment
from distutils import dir_util
from shutil import which, copyfile, rmtree
from os import path

# control wether the test runs with a database cluster or not
CLUSTER=True
pytestmark = pytest.mark.skip()

def test_one_way():
    """test the latency for a sending a vector, a matrix, and a 3D tensor
       from a simulation model to a SmartSimNode"""
    run_test("one-way")

def test_full_loop():
    """test the latency for a sending a vector, a matrix, and a 3D tensor
       from a simulation model to a SmartSimNode and then back to a model"""
    run_test("full-loop")

def test_node_sink():
    """test the latency for a sending a vector, a matrix, and a 3D tensor
       from two simulation models to a single SmartSimNode"""
    run_test("node-sink")

def run_test(test):
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
        num_packets = 5
        if test == "one-way":
            run_one_way(data_size, num_packets, test_id, cluster_size)
        elif test == "full-loop":
            run_full_loop(data_size, num_packets, test_id, cluster_size)
        else:
            run_node_sink(data_size, num_packets, test_id, cluster_size)

def run_one_way_poll_and_check(cluster_size=3):

    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/client-one-way-poll-and-check", "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict)
    experiment.create_orchestrator_cluster(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim", "sim")
    experiment.register_connection("sim", "node")

    experiment.generate(
        model_files=base_dir+"/one-way-poll-check-int/simulation.py",
        node_files=base_dir+"/one-way-poll-check-int/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)

def run_one_way(data_size, num_packets, test_id, cluster_size):
    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/client-one-way", test_id, "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets),
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict)
    experiment.create_orchestrator_cluster(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim", "sim")
    experiment.register_connection("sim", "node")

    experiment.generate(
        model_files=base_dir+"/one-way/simulation.py",
        node_files=base_dir+"/one-way/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)

def run_full_loop(data_size, num_packets, test_id, cluster_size):
    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/full-loop", test_id, "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets),
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict)
    experiment.create_orchestrator_cluster(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim", "node")
    experiment.register_connection("node", "sim")

    experiment.generate(
        model_files=base_dir+"/full-loop/simulation.py",
        node_files=base_dir+"/full-loop/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)

def run_node_sink(data_size, num_packets, test_id, cluster_size):
    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/node-sink", test_id, "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "exe_args": create_exe_args(data_size, num_packets),
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim_1 = experiment.create_model("sim_1",
                                    run_settings=sim_dict)
    sim_2 = experiment.create_model("sim_2",
                                    run_settings=sim_dict)
    experiment.create_orchestrator_cluster(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim_1", "node")
    experiment.register_connection("sim_2", "node")


    experiment.generate(
        model_files=base_dir+"/node-sink/simulation.py",
        node_files=base_dir+"/node-sink/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim_1) == "COMPLETED")
    assert(experiment.get_status(sim_2) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_one_way_poll_and_check_int():

    cluster_size = 3
    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/exp-client-one-way-poll-and-check-int", "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict)
    experiment.create_orchestrator_cluster(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim", "sim")
    experiment.register_connection("sim", "node")

    experiment.generate(
        model_files=base_dir+"/one-way-poll-check-int/simulation.py",
        node_files=base_dir+"/one-way-poll-check-int/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_one_way_poll_and_check_float():

    cluster_size = 3
    base_dir = path.dirname(path.abspath(__file__))
    experiment_dir = "".join((base_dir,"/exp-client-one-way-poll-and-check-float", "/"))
    experiment = Experiment(experiment_dir)
    alloc = experiment.get_allocation(cluster_size+2)

    node_settings = {
        "nodes": 1,
        "executable": "python node.py",
        "alloc": alloc
    }
    sim_dict = {
        "executable": "python simulation.py",
        "nodes": 1,
        "alloc": alloc
    }
    node = experiment.create_node("node",
                                  run_settings=node_settings)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict)
    experiment.create_orchestrator(alloc, db_nodes=cluster_size)
    experiment.register_connection("sim", "sim")
    experiment.register_connection("sim", "node")

    experiment.generate(
        model_files=base_dir+"/one-way-poll-check-int/simulation.py",
        node_files=base_dir+"/one-way-poll-check-int/node.py"
    )

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if os.path.isdir(experiment_dir):
        rmtree(experiment_dir)


def create_exe_args(data_size, num_packets):
    size = "--size=" + str(data_size)
    num_pack = "--num_packets=" + str(num_packets)
    return " ".join((size, num_pack))
