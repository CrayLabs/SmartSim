
# TODO write these tests
import pytest
from shutil import which
from ...launcher import SlurmLauncher

# skip if not on a slurm system
if not which("srun"):
    pytestmark = pytest.mark.skip()

def test_run_on_alloc():
    pass

def test_get_alloc():
    pass

def test_get_job_status():
    pass

def test_validate():
    pass

def test_validate_fails():
    pass

def test_free_alloc():
    pass