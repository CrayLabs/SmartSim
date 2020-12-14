from os import environ
from shutil import which

import pytest

from smartsim import slurm


def test_teardown_alloc_for_tests():
    """Not a test, just releasing the slurm allocation
    that was obtained for the tests.
    """
    if not which("srun"):
        pytest.skip()
    alloc = environ["TEST_ALLOCATION_ID"]
    slurm.release_slurm_allocation(alloc)
    del environ["TEST_ALLOCATION_ID"]
