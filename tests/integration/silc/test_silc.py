import os.path as osp
import pickle
from shutil import rmtree
import sys
import pytest

from smartsim import Experiment, constants
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings import RunSettings

"""Test SILC integration for ensembles. Two copies of the same
   program will be executed concurrently, and name collusions
   will be avoided through SILC prefixing:
   SILC will prefix each instance's tensors with a prefix
   set through environment variables by SmartSim.
"""


REDIS_PORT = 6780


try:
    import silc
    import torch
except ImportError:
    pass


@pytest.mark.skipif(('silc' not in sys.modules) or
                    ('torch' not in sys.modules),
                    reason="requires SILC and PyTorch")


def test_exchange():
    """ Run two processes, each process puts a tensor on
        the DB, then accesses the other process's tensor.
        Finally, the tensor is used to run a model.
    """

    exp = Experiment("silc_ensemble", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = RunSettings("python", "producer.py --exchange")
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
            name="producer",
            params=params,
            run_settings=rs,
            perm_strat="step"
        )

    ensemble.register_incoming_entity(ensemble.entities[0])
    ensemble.register_incoming_entity(ensemble.entities[1])

    ensemble.attach_generator_files(to_copy="./integration/silc/producer.py")

    exp.generate(ensemble, overwrite=True)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)


def test_consumer():
    """ Run three processes, each one of the first two processes
        puts a tensor on the DB; the third process accesses the 
        tensors put by the two producers.
        Finally, the tensor is used to run a model by each producer
        and the consumer accesses the two results.
    """
    exp = Experiment("silc_ensemble", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs_prod = RunSettings("python", "producer.py")
    rs_consumer = RunSettings("python", "consumer.py")
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
            name="producer",
            params=params,
            run_settings=rs_prod,
            perm_strat="step"
        )

    consumer = Model("consumer", params={}, path=ensemble.path, run_settings=rs_consumer)
    ensemble.add_model(consumer)
    print(ensemble.entities)

    ensemble.register_incoming_entity(ensemble.entities[0])
    ensemble.register_incoming_entity(ensemble.entities[1])

    ensemble.attach_generator_files(to_copy=["./integration/silc/producer.py",
                                             "./integration/silc/consumer.py"])

    exp.generate(ensemble, overwrite=True)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)
