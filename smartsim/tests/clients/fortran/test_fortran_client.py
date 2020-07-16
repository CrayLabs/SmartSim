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
                 target_names=["array_1d_unit_test"])
def test_put_get_one_dimensional_array_ftn(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["array_2d_unit_test"])
def test_put_get_two_dimensional_array_ftn(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["array_1d_unit_test"])
def test_put_get_one_dimensional_array_ftn_w_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    client_model.register_incoming_entity(client_model, 'ftn')
    client_model.enable_key_prefixing()
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["scalar_unit_test"])
def test_put_get_scalar_wo_prefixing_ftn(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["scalar_unit_test"])
def test_put_get_scalar_ftn_w_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    client_model.register_incoming_entity(client_model, 'ftn')
    client_model.enable_key_prefixing()
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["exact_key_scalar_unit_test"])
def test_put_get_scalar_exact_key_w_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.register_incoming_entity(client_model, 'ftn')
    client_model.enable_key_prefixing()
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["exact_key_scalar_unit_test"])
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["poll_key_and_check_scalar_unit_test"])
def test_poll_key_and_check_w_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.enable_key_prefixing()
    client_model.register_incoming_entity(client_model,'fortran')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["poll_exact_key_and_check_scalar_unit_test"])
def test_poll_exact_key_and_check_w_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    client_model.enable_key_prefixing()
    client_model.register_incoming_entity(client_model,'fortran')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["poll_key_and_check_scalar_unit_test"])
def test_poll_key_and_check_wo_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["poll_exact_key_and_check_scalar_unit_test"])
def test_poll_exact_key_and_check_wo_prefixing(*args, **kwargs):
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
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)
