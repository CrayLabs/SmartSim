import os
import pytest
import time
from smartsim.tests.decorators import python_client_test

from smartsim import Experiment
from distutils import dir_util
from shutil import which, copyfile, rmtree
from os import path

# control wether the test runs with a database cluster or not
CLUSTER=True

if not which("srun"):
    pytestmark = pytest.mark.skip()

base_dir = path.dirname(path.abspath(__file__))
experiment_dir = "".join((base_dir,"/python_client_test/"))
experiment = Experiment("python_client_test")
alloc = experiment.get_allocation(nodes=6)

# what size data and how many sends
# if you change this, you need to edit corresponding
# node.py and simulation.py files
data_size = "'(200,)'"
num_packets = 5

# helper function
def create_exe_args(data_size, num_packets):
    size = "--size=" + str(data_size)
    num_pack = "--num_packets=" + str(num_packets)
    return " ".join((size, num_pack))


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


@python_client_test(base_dir + "/one-way/")
def test_one_way():

    node = experiment.create_node("node",
                                  run_settings=node_settings,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    orc = experiment.create_orchestrator_cluster(alloc,
                                           path=base_dir + "/python_client_test/",
                                           overwrite=True)

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop(orchestrator=orc)


@python_client_test(base_dir + "/full-loop/")
def test_full_loop():

    node = experiment.create_node("node",
                                  run_settings=node_settings,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    orc = experiment.create_orchestrator_cluster(alloc,
                                                path=base_dir + "/python_client_test/",
                                                overwrite=True)
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop(orchestrator=orc)

@python_client_test(base_dir + "/node-sink/")
def test_node_sink():

    node = experiment.create_node("node",
                                  run_settings=node_settings,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    sim_1 = experiment.create_model("sim_1",
                                     run_settings=sim_dict,
                                     path=base_dir + "/python_client_test/",
                                     overwrite=True,
                                     enable_key_prefixing=True)
    sim_2 = experiment.create_model("sim_2",
                                     run_settings=sim_dict,
                                     path=base_dir + "/python_client_test/",
                                     overwrite=True,
                                     enable_key_prefixing=True)
    orc = experiment.create_orchestrator_cluster(alloc,
                                                 path=base_dir + "/python_client_test/",
                                                 overwrite=True)
    node.register_incoming_entity(sim_1,"python")
    node.register_incoming_entity(sim_2,"python")

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim_1) == "COMPLETED")
    assert(experiment.get_status(sim_2) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop(orchestrator=orc)


@python_client_test(base_dir + "/one-way-poll-check-int/")
def test_one_way_poll_and_check_int():

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
                                  run_settings=node_settings,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    orc = experiment.create_orchestrator_cluster(alloc,
                                                path=base_dir + "/python_client_test/",
                                                overwrite=True)

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop(orchestrator=orc)

@python_client_test(base_dir + "/one-way-poll-check-float/")
def test_one_way_poll_and_check_float():

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
                                  run_settings=node_settings,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    sim = experiment.create_model("sim",
                                  run_settings=sim_dict,
                                  path=base_dir + "/python_client_test/",
                                  overwrite=True)
    orc = experiment.create_orchestrator_cluster(alloc,
                                                path=base_dir + "/python_client_test/",
                                                overwrite=True)

    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(sim) == "COMPLETED")
    assert(experiment.get_status(node) == "COMPLETED")
    experiment.stop(orchestrator=orc)

def test_release():
    """helper to release the allocation at the end of testing"""
    experiment.release()