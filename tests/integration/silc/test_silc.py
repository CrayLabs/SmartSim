import os.path as osp
import pickle
from shutil import rmtree
import sys
import pytest

import redis

from smartsim import Experiment, constants
from smartsim.entity import Ensemble


REDIS_PORT = 6780

try:
    import silc
except ImportError:
    pass

@pytest.mark.skipif('silc' not in sys.modules,
                    reason="requires SILC")

def test_send_and_get_data():
    exp = Experiment("silc_ensemble", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = exp.create_orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = {"executable": "python", "exe_args": "model_methods_torch.py"}
    params = {"mult": [-1, 1]}
    ensemble = Ensemble(
            name="example",
            params=params,
            run_settings=rs,
            perm_strat="step"
        )

    ensemble.entities[0].register_incoming_entity(ensemble.entities[1], 'python')
    ensemble.entities[1].register_incoming_entity(ensemble.entities[0], 'python')
    ensemble.attach_generator_files(to_copy="./test_configs/model_methods_torch.py")

    exp.generate(ensemble, overwrite=True)

    # start the models
    exp.start(ensemble, summary=True)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)
