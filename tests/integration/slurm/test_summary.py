import pytest
import os.path as osp
from os import environ
from shutil import rmtree, which

from smartsim import slurm
from smartsim import Experiment


if not which("srun"):
    pytestmark = pytest.mark.skip

def test_summary():

    exp = Experiment("summary_test", launcher="slurm")

    alloc = slurm.get_slurm_allocation(nodes=2)

    run_settings = {
        "executable": "python",
        "exe_args": "sleep.py",
        "nodes": 1,
        "alloc": alloc,
    }
    bad_run_settings = run_settings.copy()
    bad_run_settings["exe_args"] = "bad.py"

    # generate one model that will fail and one that will succeed
    sleep_model = exp.create_model("sleep", run_settings)
    sleep_model.attach_generator_files(to_copy="./test_configs/sleep.py")
    bad_model = exp.create_model("bad", bad_run_settings)
    bad_model.attach_generator_files(to_copy="./test_configs/bad.py")

    exp.generate(sleep_model, bad_model)

    # start and poll
    exp.start(sleep_model, bad_model)
    exp.poll()

    summary_df = exp.summary()
    print(summary_df)
    row = summary_df.loc[1]

    assert sleep_model.name == row["Name"]
    assert sleep_model.type == row["Entity-Type"]
    assert 0 == int(row["RunID"])
    assert 0 == int(row["Returncode"])
    assert None == row["Error"]
    assert None == row["Output"]

    row_1 = summary_df.loc[0]

    assert bad_model.name == row_1["Name"]
    assert bad_model.type == row_1["Entity-Type"]
    assert 0 == int(row_1["RunID"])
    assert 0 != int(row_1["Returncode"])
    assert row_1["Error"]

    if osp.isdir(exp.exp_path):
        rmtree(exp.exp_path)

    slurm.release_slurm_allocation(alloc)


def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id
