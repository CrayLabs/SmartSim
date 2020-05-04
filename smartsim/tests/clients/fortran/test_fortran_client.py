import pytest
from os import path
from smartsim.tests.clients.client_test_shell import compiled_client_test_builder

pytestmark = pytest.mark.skip()

def test_put_get_fortran():
    """ This funtion tests putting and getting
        1D and 2D arrays from database.
    """
    test_path = path.dirname(path.abspath(__file__))
    assert(compiled_client_test_builder(test_path, "client_tester", ""))
