import time

import pytest

from smartsim import slurm
from smartsim.error import AllocationError

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Test is only for Slurm WLM systems")


def test_get_release_allocation():
    """test slurm interface for obtaining allocations"""
    alloc = slurm.get_allocation(nodes=1, time="00:05:00")
    time.sleep(5)  # give slurm a rest
    slurm.release_allocation(alloc)


def test_get_release_allocation_w_options():
    """test slurm interface for obtaining allocations"""
    options = {"ntasks-per-node": 1}
    alloc = slurm.get_allocation(nodes=1, time="00:05:00", options=options)
    time.sleep(5)  # give slurm a rest
    slurm.release_allocation(alloc)


# --------- Error handling ----------------------------


def test_release_non_existant_alloc():
    """Release allocation that doesn't exist"""
    with pytest.raises(AllocationError):
        slurm.release_allocation(00000000)
