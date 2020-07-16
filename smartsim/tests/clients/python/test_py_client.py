import pytest
from shutil import which
from os import path
from smartsim.tests.decorators import python_client_test
from smartsim import Experiment

# control wether the test runs with a database cluster or not
CLUSTER=True

if not which("srun"):
    pytestmark = pytest.mark.skip()

test_dir=path.dirname(path.abspath(__file__)) + "/"
alloc_experiment = Experiment("alloc_retrieval")
alloc = alloc_experiment.get_allocation(nodes=5, ppn=1)

@python_client_test()
def test_ndarray_put_get_wo_prefixing():
    """ This function tests putting and getting one-, two-,
        and three-dimensional ndarrays to and from the SmartSim
        database.  Success is based on equality of the sent
        and retreived arrays. All supported array data types are
        tested herein.  Key prefixing is not used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "ndarray_put_get.py",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+'/ndarray_put_get/ndarray_put_get.py'])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_ndarray_put_get_w_prefixing():
    """ This function tests putting and getting one-, two-,
        and three-dimensional ndarrays to and from the SmartSim
        database.  Success is based on equality of the sent
        and retreived arrays. All supported array data types are
        tested herein.  Key prefixing is used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "ndarray_put_get.py "\
                                "--source=client_test",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+'/ndarray_put_get/ndarray_put_get.py'])
    client_model.register_incoming_entity(client_model, 'python')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_ndarray_exact_key_put_get_wo_prefixing():
    """ This function tests putting and getting one-, two-,
        and three-dimensional ndarrays to and from the SmartSim
        database.  Success is based on equality of the sent
        and retreived arrays. All supported array data types are
        tested herein.  Exact key put and get methods are used.
        Key prefixing is not used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "ndarray_exact_key_put_get.py",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/ndarray_exact_key_put_get/"\
                          "ndarray_exact_key_put_get.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_ndarray_exact_key_put_get_w_prefixing():
    """ This function tests putting and getting one-, two-,
        and three-dimensional ndarrays to and from the SmartSim
        database.  Success is based on equality of the sent
        and retreived arrays. All supported array data types are
        tested herein.  Exact key put and get methods are used.
        Key prefixing is used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "ndarray_exact_key_put_get.py "\
                                "--source=client_test",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/ndarray_exact_key_put_get/"\
                          "ndarray_exact_key_put_get.py"])
    client_model.register_incoming_entity(client_model, 'python')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_scalar_put_get_wo_prefixing_no_cluster():
    """ This function tests putting and getting scalars to and
        from the SmartSim database.  Success is based on equality
        of the sent and retreived scalars. All supported scalar
        data types are tested herein.  Key prefixing is not used
        in this test.  A single database node is used in this
        test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "scalar_put_get_no_cluster.py ",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_put_get_no_cluster/"\
                          "scalar_put_get_no_cluster.py"])
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_scalar_put_get_wo_prefixing():
    """ This function tests putting and getting scalars to and
        from the SmartSim database.  Success is based on equality
        of the sent and retreived scalars. All supported scalar
        data types are tested herein.  Key prefixing is not used
        in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "scalar_put_get.py ",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_put_get/"\
                          "scalar_put_get.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_scalar_put_get_w_prefixing():
    """ This function tests putting and getting scalars to and
        from the SmartSim database.  Success is based on equality
        of the sent and retreived scalars. All supported scalar
        data types are tested herein.  Key prefixing is used
        in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "scalar_put_get.py "\
                                "--source=client_test",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_put_get/"\
                          "scalar_put_get.py"])
    client_model.register_incoming_entity(client_model, 'python')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_scalar_exact_key_put_get_wo_prefixing():
    """ This function tests putting and getting scalars to and
        from the SmartSim database.  Success is based on equality
        of the sent and retreived scalars. All supported scalar
        data types are tested herein.  Exact key placement
        without key prefixing activated is used.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "scalar_exact_key_put_get.py ",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_exact_key_put_get/"\
                          "scalar_exact_key_put_get.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_scalar_exact_key_put_get_w_prefixing():
    """ This function tests putting and getting scalars to and
        from the SmartSim database.  Success is based on equality
        of the sent and retreived scalars. All supported scalar
        data types are tested herein.  Exact key placement
        with key prefixing activated is used.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 1,
                    "executable":"python",
                    "exe_args": "scalar_exact_key_put_get.py "\
                                "--source=client_test",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                           run_settings=run_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_exact_key_put_get/"\
                          "scalar_exact_key_put_get.py"])
    client_model.register_incoming_entity(client_model, 'python')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_poll_check_key_wo_prefixing():
    """ This function tests putting a scalar into the database
        and polling and checking for the key and value
        in a separate SmartSim node.  Success is based
        sending and polling and checking successfully.
        All supported scalar data types are tested herein.
        Key prefixing is not used.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "sender.py ",
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "receiver.py ",
                            "alloc": alloc}
    client_model = experiment.create_model("client_sender",
                                           run_settings=send_client_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check/sender.py"])
    client_node = experiment.create_node("client_node",
                                         run_settings=recv_client_settings)
    client_node.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check/receiver.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    assert(experiment.get_status(client_node) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_poll_check_key_w_prefixing():
    """ This function tests putting a scalar into the database
        and polling and checking for the key and value
        in a separate SmartSim node.  Success is based
        sending and polling and checking successfully.
        All supported scalar data types are tested herein.
        Key prefixing is used.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "sender.py",
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "receiver.py "\
                                        "--source=client_sender",
                            "alloc": alloc}
    client_model = experiment.create_model("client_sender",
                                           run_settings=send_client_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check/sender.py"])
    client_node = experiment.create_node("client_node",
                                         run_settings=recv_client_settings)
    client_node.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check/receiver.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    client_node.register_incoming_entity(client_model, 'python')
    client_node.enable_key_prefixing()
    client_model.enable_key_prefixing()
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    assert(experiment.get_status(client_node) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_poll_check_exact_key_wo_prefixing():
    """ This function tests putting a scalar into the database
        and polling and checking for the key and value
        in a separate SmartSim node.  Success is based
        sending and polling and checking successfully.
        All supported scalar data types are tested herein.
        Exact key (no prefixing) methods are used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "sender.py",
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "receiver.py",
                            "alloc": alloc}
    client_model = experiment.create_model("client_sender",
                                           run_settings=send_client_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check_exact_key/sender.py"])
    client_node = experiment.create_node("client_node",
                                         run_settings=recv_client_settings)
    client_node.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check_exact_key/receiver.py"])
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    assert(experiment.get_status(client_node) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

@python_client_test()
def test_poll_check_exact_key_w_prefixing():
    """ This function tests putting a scalar into the database
        and polling and checking for the key and value
        in a separate SmartSim node.  Success is based
        sending and polling and checking successfully.
        All supported scalar data types are tested herein.
        Exact key (no prefixing) methods are used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "sender.py",
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 1,
                            "executable":"python",
                            "exe_args": "receiver.py "\
                                        "--source=client_sender",
                            "alloc": alloc}
    client_model = experiment.create_model("client_sender",
                                           run_settings=send_client_settings)
    client_model.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check_exact_key/sender.py"])
    client_node = experiment.create_node("client_node",
                                         run_settings=recv_client_settings)
    client_node.attach_generator_files(
        to_copy=[test_dir+"/scalar_poll_and_check_exact_key/receiver.py"])
    client_node.register_incoming_entity(client_model, 'python')
    client_node.enable_key_prefixing()
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=3, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    assert(experiment.get_status(client_node) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(poll_db=True)

def test_release():
    """helper to release the allocation at the end of testing"""
    alloc_experiment.release()
