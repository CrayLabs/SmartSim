import pytest
from os import environ
from shutil import which
from smartsim import slurm


def test_setup_alloc_for_tests():
   """Not actually a test. Sets up the allocation on slurm
      for all the slurm controller tests.
   """
   if not which("srun"):
      pytest.skip()
   alloc = slurm.get_slurm_allocation(nodes=5, ppn=3)
   environ["TEST_ALLOCATION_ID"] = str(alloc)
   assert("TEST_ALLOCATION_ID" in environ)

