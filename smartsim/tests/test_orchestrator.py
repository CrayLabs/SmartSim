from smartsim import Experiment
from os import path, getcwd
from ..error import SmartSimError
import pytest
from smartsim.tests.decorators import orchestrator_test_slurm
from smartsim.tests.decorators import orchestrator_test_local
import filecmp

test_path = path.join(getcwd(),  "./orchestrator_test/")

# --- database reconnect ----------------------------------------------

@orchestrator_test_local
def test_db_reconnect_send_receive_local():

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    sim_dict = {"executable": "python reconnect_sim.py" }
    exp_1 = Experiment("exp_1", launcher="local")
    O1 = exp_1.create_orchestrator(path=exp_1_dir)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()

    exp_2 = Experiment("exp_2", launcher="local")
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)

@orchestrator_test_slurm
def test_db_cluster_reconnect_status_and_stop():

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    alloc_1 = exp_1.get_allocation(nodes=5, ppn=3, duration="00:10:00")
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": alloc_1,
        "exe_args":"--cluster"
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir,
                                   db_nodes=3, dpn=3, alloc=alloc_1)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.stop(orchestrator=O2)
    exp_2.poll(poll_db=True)
    assert("RUNNING" not in exp_2.get_status(O2))

    exp_1.release(alloc_id=alloc_1)
    exp_2.release(alloc_id=alloc_2)

@orchestrator_test_slurm
def test_db_reconnect_status_and_stop():

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    alloc_1 = exp_1.get_allocation(nodes=2, ppn=1, duration="00:10:00")
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": alloc_1
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, alloc=alloc_1)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.stop(orchestrator=O2)
    exp_2.poll(poll_db=True)
    assert("RUNNING" not in exp_2.get_status(O2))

    exp_1.release(alloc_id=alloc_1)
    exp_2.release(alloc_id=alloc_2)

@orchestrator_test_slurm
def test_db_cluster_reconnect_send_receive():
    """test that a database can be reconnected to and data can be
    retrieved
    """

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    alloc_1 = exp_1.get_allocation(nodes=5, ppn=3, duration="00:10:00")
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": alloc_1,
        "exe_args":"--cluster"
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir,
                                   db_nodes=3, dpn=3, alloc=alloc_1)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python reconnect_node.py",
        "alloc": alloc_2,
        "exe_args":"--cluster"
    }
    N2 = exp_2.create_node("N2", path=exp_2_dir, run_settings=node_settings)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.start(ssnodes=[N2])
    exp_2.poll()

    f1 = "/".join((exp_1_dir,"M1.out"))
    f2 = "/".join((exp_2_dir,"N2.out"))
    num_lines = sum(1 for line in open(f1))
    assert(num_lines>0)
    assert(filecmp.cmp(f1,f2))

    exp_2.stop(orchestrator=O2)
    exp_1.release(alloc_id=alloc_1)
    exp_2.release(alloc_id=alloc_2)

@orchestrator_test_slurm
def test_db_reconnect_send_receive():
    """test that a database can be reconnected to and data can be
    retrieved
    """

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    alloc_1 = exp_1.get_allocation(nodes=2, ppn=1, duration="00:10:00")
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": alloc_1
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir, alloc=alloc_1)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    node_settings = {
        "nodes": 1,
        "executable": "python reconnect_node.py",
        "alloc": alloc_2
    }
    N2 = exp_2.create_node("N2", path=exp_2_dir, run_settings=node_settings)
    assert(all(status == "RUNNING" for status in exp_2.get_status(O2)))
    exp_2.start(ssnodes=[N2])
    exp_2.poll()

    f1 = "/".join((exp_1_dir,"M1.out"))
    f2 = "/".join((exp_2_dir,"N2.out"))
    num_lines = sum(1 for line in open(f1))
    assert(num_lines>0)
    assert(filecmp.cmp(f1,f2))

    exp_2.stop(orchestrator=O2)
    exp_1.release(alloc_id=alloc_1)
    exp_2.release(alloc_id=alloc_2)

# --- error handling ---------------------------------------------------

@orchestrator_test_slurm
def test_db_reconnection_not_running_failure():

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    exp_1 = Experiment("exp_1")
    alloc_1 = exp_1.get_allocation(nodes=5, ppn=3, duration="00:10:00")
    sim_dict = {
        "executable": "python reconnect_sim.py",
        "nodes": 1,
        "alloc": alloc_1,
        "exe_args":"--cluster"
    }
    O1 = exp_1.create_orchestrator(path=exp_1_dir,
                                   db_nodes=3, dpn=3, alloc=alloc_1)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()
    exp_1.poll()
    assert("FAILED" not in exp_1.get_status(M1))
    exp_1.stop(orchestrator=O1)
    exp_1.poll(poll_db=True)

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)

    exp_1.release(alloc_id=alloc_1)
    exp_2.release(alloc_id=alloc_2)

@orchestrator_test_slurm
def test_db_reconnection_no_file():

    exp_1_dir = "/".join((test_path,"exp_1"))

    exp_2 = Experiment("exp_2")
    alloc_2 = exp_2.get_allocation(nodes=1, ppn=1)
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)
    exp_2.release(alloc_id=alloc_2)
