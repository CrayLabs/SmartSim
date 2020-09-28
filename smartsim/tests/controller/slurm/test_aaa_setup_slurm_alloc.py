import pytest
from os import environ
from shutil import which
from smartsim.control import Controller


def test_setup_alloc_for_tests():
   """Not actually a test. Sets up the allocation on slurm
      for all the slurm controller tests.
   """
   if not which("srun"):
      pytest.skip()
   ctrl = Controller()
   alloc = ctrl.get_allocation(nodes=5, ppn=3)
   environ["TEST_ALLOCATION_ID"] = str(alloc)
   assert("TEST_ALLOCATION_ID" in environ)

