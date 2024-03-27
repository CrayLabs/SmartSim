import pytest

from smartsim._core.utils.network import find_free_port

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_find_free_port_no_start():
    """Test that a free port is identified and returned when no
    starting port number is specified"""
    port = find_free_port()
    assert port > 0


@pytest.mark.parametrize(
    "start_at",
    [
        pytest.param(1000, id="start at 1000"),
        pytest.param(2000, id="start at 2000"),
        pytest.param(5000, id="start at 5000"),
        pytest.param(10000, id="start at 10000"),
        pytest.param(16000, id="start at 16000"),
    ],
)
def test_find_free_port_range_specified(start_at):
    """Test that a free port greater than or equal to the specified
    starting port number is identified and returned"""
    port = find_free_port(start_at)
    assert port >= start_at
