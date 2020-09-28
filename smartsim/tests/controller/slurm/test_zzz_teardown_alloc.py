import pytest
from os import environ
from shutil import which
from smartsim.control import Controller

def test_teardown_alloc_for_tests():
   """Not a test, just releasing the slurm allocation
      that was obtained for the tests.
   """
   if not which("srun"):
      pytest.skip()
   ctrl = Controller()
   alloc = environ["TEST_ALLOCATION_ID"]
   ctrl.add_allocation(alloc_id=alloc)
   ctrl.release(alloc_id=alloc)
   del environ["TEST_ALLOCATION_ID"]