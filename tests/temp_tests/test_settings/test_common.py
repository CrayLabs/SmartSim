import pytest

from smartsim.settings.common import set_check_input


def test_check_set_raise_error():
    with pytest.raises(TypeError):
        set_check_input(key="test", value=3)
    with pytest.raises(TypeError):
        set_check_input(key=3, value="str")
    with pytest.raises(TypeError):
        set_check_input(key=2, value=None)
