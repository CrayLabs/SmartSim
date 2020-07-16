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
                 target_names=["client_test_scalar_put_get"])
def test_put_get_scalar(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_array_1d_put_get"])
def test_put_get_array_1d(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_array_2d_put_get"])
def test_put_get_array_2d(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_poll_key_and_check"])
def test_poll_key_and_check(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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

# Exact key tests
@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_exact_key_scalar_put_get"])
def test_exact_key_put_get_scalar(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_exact_key_array_1d_put_get"])
def test_exact_key_put_get_array_1d(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_exact_key_array_2d_put_get"])
def test_exact_key_put_get_array_2d(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
                 target_names=["client_test_poll_exact_key_and_check"])
def test_exact_key_poll_key_and_check(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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

# Exact key tests wo prefixing
@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_exact_key_scalar_put_get"])
def test_exact_key_put_get_scalar_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
    client_model.register_incoming_entity(client_model,'c')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_exact_key_array_1d_put_get"])
def test_exact_key_put_get_array_1d_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
    client_model.register_incoming_entity(client_model,'c')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_exact_key_array_2d_put_get"])
def test_exact_key_put_get_array_2d_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
    client_model.register_incoming_entity(client_model,'c')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)

@compiled_client_test(test_dir=test_dir,
                 target_names=["client_test_poll_exact_key_and_check"])
def test_exact_key_poll_key_and_check_w_prefixing(*args, **kwargs):
    """ This function tests putting and getting a scalar
        to and from the SmartSim database.  Success is
        based on equality of the sent and retreived scalar
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
    client_model.register_incoming_entity(client_model,'c')
    orc = experiment.create_orchestrator(db_nodes=1, alloc=alloc)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop(orchestrator=orc)
    experiment.poll(interval=1, poll_db=True)
