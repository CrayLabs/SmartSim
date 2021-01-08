import os
from shutil import which

import pytest

from smartsim import constants
from smartsim.error.errors import LauncherError
from smartsim.launcher import SlurmLauncher
from smartsim.launcher.stepInfo import SlurmStepInfo

# skip if not on a slurm system
if not which("srun"):
    pytestmark = pytest.mark.skip()

slurm = SlurmLauncher()

def test_get_step_status():
    """test calling get_step_status for step that doesnt exist"""
    status = slurm.get_step_status(11111.1)
    assert isinstance(status, SlurmStepInfo)
    assert status.status == constants.STATUS_FAILED
    assert status.returncode == "NAN"
    assert status.error == None
    assert status.output == None


def test_bad_get_step_nodes():
    """test call of get_step_nodes with a step that doesnt exist"""
    with pytest.raises(LauncherError):
        slurm.get_step_nodes(111111.1)


def test_no_alloc_create_step():
    """create_step called without an alloc in run_settings"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    run_settings = {
        "nodes": 1,
        "out_file": cwd + "/out.txt",
        "err_file": cwd + "/err.txt",
        "cwd": cwd,
        "executable": "a.out",
        "exe_args": "--input",
    }
    with pytest.raises(LauncherError):
        slurm.create_step("test", run_settings)


@pytest.mark.skip(
    reason="We currently don't test that a user's inputted allocation exists"
)
def test_bad_alloc_create_step():
    """create_step called with a non-existant allocation"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    run_settings = {
        "nodes": 1,
        "ppn": 1,
        "out_file": cwd + "/out.txt",
        "err_file": cwd + "/err.txt",
        "cwd": cwd,
        "executable": "a.out",
        "exe_args": "--input",
        "alloc": 1111111,
    }
    with pytest.raises(LauncherError):
        slurm.create_step("test", run_settings)
