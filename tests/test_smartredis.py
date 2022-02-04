import pytest

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model

"""Test smartredis integration for ensembles. Two copies of the same
   program will be executed concurrently, and name collisions
   will be avoided through smartredis prefixing:
   smartredis will prefix each instance's tensors with a prefix
   set through environment variables by SmartSim.
"""


REDIS_PORT = 6780

shouldrun = True
try:
    import torch

    import smartredis
except ImportError:
    shouldrun = False

torch_available = "torch" in installed_redisai_backends()

shouldrun &= torch_available

pytestmark = pytest.mark.skipif(
    not shouldrun,
    reason="requires PyTorch, SmartRedis, and RedisAI's Torch backend",
)


def test_exchange(fileutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.
    """

    test_dir = fileutils.make_test_dir("smartredis_ensemble_exchange_test")
    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = exp.create_run_settings("python", "producer.py --exchange")
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


def test_consumer(fileutils):
    """Run three processes, each one of the first two processes
    puts a tensor on the DB; the third process accesses the
    tensors put by the two producers.
    Finally, the tensor is used to run a model by each producer
    and the consumer accesses the two results.
    """
    test_dir = fileutils.make_test_dir("smartredis_ensemble_consumer_test")
    exp = Experiment(
        "smartredis_ensemble_consumer", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs_prod = exp.create_run_settings("python", "producer.py")
    rs_consumer = exp.create_run_settings("python", "consumer.py")
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
