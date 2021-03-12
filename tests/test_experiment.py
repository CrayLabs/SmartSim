import pytest

from smartsim import Experiment

def test_bad_exp_path():
    with pytest.raises(NotADirectoryError):
        exp = Experiment("test", "not-a-directory")

def test_type_exp_path():
    with pytest.raises(TypeError):
        exp = Experiment("test", ["this-is-a-list-dummy"])
