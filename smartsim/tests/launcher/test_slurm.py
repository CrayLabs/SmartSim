
# TODO write these tests
import pytest
from shutil import which
from ...launcher import SlurmLauncher

# skip if not on a slurm system
if not which("srun"):
    pytestmark = pytest.mark.skip()

def test_run():
    pass

def test_run_on_alloc():
    pass

def test_run_script():
    pass

def test_submit_and_forget():
    pass
