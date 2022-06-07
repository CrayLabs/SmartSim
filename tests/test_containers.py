import pytest
from shutil import which

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings.containers import Singularity

REDIS_PORT = 6780

# Check if singularity is available as command line tool
singularity_exists = which('singularity') is not None

@pytest.mark.skipif(True)
def test_singularity_redis(fileutils):
    """Run two processes in singularity containers, each process puts a tensor
    on the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.
    """

    test_dir = fileutils.make_test_dir("test_singularity_redis")
    exp = Experiment(
        "test_singularity_redis", exp_path=test_dir, launcher="auto"
    )

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    # TODO: Use CrayLabs image hosted on dockerhub
    container = Singularity('docker://benalbrecht10/smartsim-testing')
    rs = exp.create_run_settings("python", "send_data.py", container=container)
    model = exp.create_model('send_data', rs)
    model.attach_generator_files(to_copy='test_configs/send_data.py')
    exp.generate(model, overwrite=True)
    exp.start(model, block=True, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(model)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # model experiment failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_exchange(fileutils, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.
    """

    test_dir = fileutils.make_test_dir()
    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity('docker://benalbrecht10/smartsim-testing:latest')

    rs = exp.create_run_settings("python", "producer.py --exchange", container=container)
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer",
        params=params,
        run_settings=rs,
        perm_strat="step",
    )

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_consumer(fileutils, wlmutils):
    """Run three processes, each one of the first two processes
    puts a tensor on the DB; the third process accesses the
    tensors put by the two producers.
    Finally, the tensor is used to run a model by each producer
    and the consumer accesses the two results.
    """
    test_dir = fileutils.make_test_dir()
    exp = Experiment(
        "smartredis_ensemble_consumer", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity('docker://benalbrecht10/smartsim-testing:latest')

    rs_prod = exp.create_run_settings("python3", "producer.py", container=container)
    rs_consumer = exp.create_run_settings("python3", "consumer.py", container=container)
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer", params=params, run_settings=rs_prod, perm_strat="step"
    )

    consumer = Model(
        "consumer", params={}, path=ensemble.path, run_settings=rs_consumer
    )
    ensemble.add_model(consumer)

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())
