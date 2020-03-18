import pytest
from os import path
from smartsim.tests.clients.client_test_shell import compiled_client_test_builder

def test_put_get_one_dimensional_array_cpp():
    """ This funtion tests putting a one dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """
    test_path = path.dirname(path.abspath(__file__))
    assert(compiled_client_test_builder(test_path, "client_tester_1D", "1000"))

def test_put_get_two_dimensional_array_cpp():
    """ This funtion tests putting a two dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """
    test_path = path.dirname(path.abspath(__file__))
    assert(compiled_client_test_builder(test_path, "client_tester_2D", "1000"))

def test_put_get_three_dimensional_array_cpp():
    """ This funtion tests putting a three dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """
    test_path = path.dirname(path.abspath(__file__))
    assert(compiled_client_test_builder(test_path, "client_tester_3D", "100"))
