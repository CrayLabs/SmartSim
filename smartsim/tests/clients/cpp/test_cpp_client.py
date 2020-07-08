import pytest
from smartsim.tests.decorators import compiled_client_test
from smartsim import Experiment
from shutil import which
from os import path

if not which("srun"):
    pytestmark = pytest.mark.skip()

test_dir=path.dirname(path.abspath(__file__))

alloc_experiment = Experiment("alloc_retrieval")
alloc = alloc_experiment.get_allocation(nodes=5, ppn=2)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_1D"])
def test_put_get_one_dimensional_array_cpp(*args, **kwargs):
    """ This function tests putting and getting a one dimensional
        array to and from the SmartSim database.  Success is
        based on equality of the sent and retreived arrays.
        All supported array data types are tested herein.
    """

    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "10000",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_2D"])
def test_put_get_two_dimensional_array_cpp(*args, **kwargs):
    """ This function tests putting and getting a two dimensional
        array to and from the SmartSim database.  Success is
        based on equality of the sent and retreived arrays.
        All supported array data types are tested herein.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "1000",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_3D"])
def test_put_get_three_dimensional_array_cpp(*args, **kwargs):
    """ This function tests putting and getting a three dimensional
        array to and from the SmartSim database.  Success is
        based on equality of the sent and retreived arrays.
        All supported array data types are tested herein.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "100",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_1D"])
def test_put_get_one_dimensional_array_cpp_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a one dimensional
        array to and from the SmartSim database.  Success is
        based on equality of the sent and retreived arrays.
        All supported array data types are tested herein.
        Key prefixing is used in this test.
    """

    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "10000",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    client_model.register_incoming_entity(client_model, 'cpp')
    client_model.enable_key_prefixing()
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_scalar_put_get"])
def test_put_get_scalar_cpp(*args, **kwargs):
    """ This function tests putting and getting a scalar
        value to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalars.
        All supported scalar data types are tested herein.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_scalar_put_get"])
def test_put_get_scalar_cpp_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        value to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalars.
        All supported scalar data types are tested herein.
        Key prefixing is used in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    client_model.register_incoming_entity(client_model, 'cpp')
    client_model.enable_key_prefixing()
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_env_parsing"])
def test_env_parsing(*args, **kwargs):
    """ This function tests the environment variable
        parsing function for SSKEYIN and SSKEYOUT
        that sets key prefixes.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.register_incoming_entity(client_model, 'cpp')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_scalar_put_get_exact_keys"])
def test_put_get_scalar_exact_key_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        value to and from the SmartSim database using an
        exact key.  Success is based on equality of the sent
        and retreived scalars. All supported scalar data types
        are tested herein.  Key prefixing is enabled in this
        test, and the exact key is meant to disregard the
        prefix.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.register_incoming_entity(client_model, 'cpp')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_scalar_put_get_exact_keys"])
def test_put_get_scalar_exact_key_wo_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        value to and from the SmartSim database using an
        exact key.  Success is based on equality of the sent
        and retreived scalars. All supported scalar data types
        are tested herein.  Key prefixing is not used
        in this test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_3D_exact_key"])
def test_put_get_array_exact_key_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a 3D array
        to and from the SmartSim database using an
        exact key.  Success is based on equality of the sent
        and retreived array. All supported array data types
        are tested herein.  Key prefixing is enabled in this
        test, and the exact key is meant to disregard the
        prefix.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "100",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.register_incoming_entity(client_model, 'cpp')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_array_put_get_3D_exact_key"])
def test_put_get_array_exact_key_wo_prefixing(*args, **kwargs):
    """ This function tests putting and getting a 3D array
        to and from the SmartSim database using an
        exact key.  Success is based on equality of the sent
        and retreived array. All supported array data types
        are tested herein.  Key prefixing is disabled in this
        test.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "100",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_poll_check_scalar_send",
                               "client_test_poll_check_scalar_receive"])
def test_poll_check_scalar(*args, **kwargs):
    """ This function tests polling and checking for scalar values.
        Success is based on the receiving client being able to find
        the key and exact scalar value in the database.  Key prefixing
        is disabled in this text.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][0],
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][1],
                            "alloc": alloc}
    client_send_model = experiment.create_model("client_send",
        run_settings=send_client_settings)
    client_recv_model = experiment.create_model("client_recv",
        run_settings=recv_client_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_send_model) == "COMPLETED")
    assert(experiment.get_status(client_recv_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_poll_check_exact_keys_scalar_send",
                               "client_test_poll_check_exact_keys_scalar_receive"])
def test_poll_check_exact_key_scalar_wo_prefixing(*args, **kwargs):
    """ This function tests polling and checking for scalar values
        using exact keys.  Success is based on the receiving client
        being able to find the key and exact scalar value in the database.
        Key prefixing is disabled in this text.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][0],
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][1],
                            "alloc": alloc}
    client_send_model = experiment.create_model("client_send",
        run_settings=send_client_settings)
    client_recv_model = experiment.create_model("client_recv",
        run_settings=recv_client_settings)
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_send_model) == "COMPLETED")
    assert(experiment.get_status(client_recv_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                target_names=["client_test_poll_check_exact_keys_scalar_send",
                               "client_test_poll_check_exact_keys_scalar_receive"])
def test_poll_check_exact_key_scalar_w_prefixing(*args, **kwargs):
    """ This function tests polling and checking for scalar values
        using exact keys.  Success is based on the receiving client
        being able to find the key and exact scalar value in the database.
        Key prefixing is enabled in this text.
    """
    experiment = Experiment("client_test")
    experiment.add_allocation(alloc)
    send_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][0],
                            "alloc": alloc}
    recv_client_settings = {"nodes":1,
                            "ppn": 2,
                            "executable":kwargs['binary_names'][1],
                            "alloc": alloc}
    client_send_model = experiment.create_model("client_send",
        run_settings=send_client_settings)
    client_recv_model = experiment.create_model("client_recv",
        run_settings=recv_client_settings)
    client_recv_model.register_incoming_entity(client_send_model, 'cpp')
    client_send_model.enable_key_prefixing()
    client_recv_model.enable_key_prefixing()
    orc = experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_send_model) == "COMPLETED")
    assert(experiment.get_status(client_recv_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

def test_release():
    """Release the allocation used to run all these tests"""
    alloc_experiment.release()