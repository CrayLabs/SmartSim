
import os.path as osp
import pickle
from shutil import rmtree

import redis

from smartsim import Experiment, constants

REDIS_PORT = 6780

def _get_data(name, iters):
    client = redis.Redis(host="localhost", port=REDIS_PORT)
    data = []
    for i in range(iters):
        key = "key_" + name + "_" + str(i)
        serialized = client.get(key)
        data.append(pickle.loads(serialized))
    assert(len(data) == 10)
    return data

def test_send_and_get_data():
    exp = Experiment("send_get_data", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create and start a database
    orc = exp.create_orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = {
        "executable": "python",
        "exe_args": "send_data.py --iters 10 --name dg1"
    }
    rs_2 = {
        "executable": "python",
        "exe_args": "send_data.py --iters 10 --name dg2"
    }
    model_dg1 = exp.create_model("dg1", run_settings=rs)
    model_dg2 = exp.create_model("dg2", run_settings=rs_2)

    # attach file to run and generate files
    model_dg1.attach_generator_files(to_copy="./test_configs/send_data.py")
    model_dg2.attach_generator_files(to_copy="./test_configs/send_data.py")
    exp.generate(model_dg1, model_dg2, overwrite=True)

    # start the models
    exp.start(model_dg1, model_dg2, summary=True)

    # get and confirm statuses
    statuses = exp.get_status(model_dg1, model_dg2)
    assert(all([stat == constants.STATUS_COMPLETED for stat in statuses]))

    # get the data generated for both models
    # and make sure the data looks right
    dg1_data = _get_data(model_dg1.name, 10)
    dg2_data = _get_data(model_dg2.name, 10)
    shape = (5000,)
    assert(all([d.shape == shape for d in dg1_data]))
    assert(all([d.shape == shape for d in dg2_data]))

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)