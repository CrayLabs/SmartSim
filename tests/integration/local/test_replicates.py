"""
Test the running of replicate models with the local launcher
"""
import os.path as osp
from shutil import rmtree

from smartsim import Experiment, constants


def test_replicates():
    exp = Experiment("replicates-local", launcher="local")

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    # create some models with an ensemble
    run_settings = {"executable": "python", "exe_args": "sleep.py --time 10"}
    models = []
    for i in range(6):
        model = exp.create_model(f"Model_{i}", run_settings)
        model.attach_generator_files(to_copy="./test_configs/sleep.py")
        models.append(model)

    # generate file structure
    exp.generate(*models)

    # start the experiment
    exp.start(*models, block=True, summary=True)

    # get and confirm statuses
    statuses = exp.get_status(*models)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])

    # get summary and confirm exit_codes
    summary = exp.summary()
    for i in range(6):
        row = summary.loc[i]
        assert int(row["Returncode"]) == 0

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)
