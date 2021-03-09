import os.path as osp
import pickle
from shutil import rmtree
import sys
import pytest

import redis

from smartsim import Experiment, constants
from smartsim.entity import Ensemble, Model


REDIS_PORT = 6780

try:
    import silc
except ImportError:
    pass

@pytest.mark.skipif('silc' not in sys.modules,
                    reason="requires SILC")

def test_exchange():
    exp = Experiment("silc_ensemble", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = exp.create_orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = {"executable": "python", "exe_args": "producer.py --exchange"}
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
    exp = Experiment("silc_ensemble", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = exp.create_orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs_prod = {"executable": "python", "exe_args": "producer.py"}
    rs_consumer = {"executable": "python", "exe_args": "consumer.py"}
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
