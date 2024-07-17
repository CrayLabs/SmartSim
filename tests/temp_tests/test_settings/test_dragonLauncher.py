import pytest

from smartsim._core.launcher.dragon.dragonLauncher import _as_run_request_view
from smartsim._core.schemas.dragonRequests import DragonRunRequestView
from smartsim.settings import LaunchSettings
from smartsim.settings.builders.launch.dragon import DragonArgBuilder
from smartsim.settings.launchCommand import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Dragon)
    assert ls.launch_args.launcher_str() == LauncherType.Dragon.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nodes", id="set_nodes"),
        pytest.param(
            "set_tasks_per_node", (2,), "2", "tasks_per_node", id="set_tasks_per_node"
        ),
    ],
)
def test_dragon_class_methods(function, value, flag, result):
    dragonLauncher = LaunchSettings(launcher=LauncherType.Dragon)
    assert isinstance(dragonLauncher._arg_builder, DragonArgBuilder)
    getattr(dragonLauncher.launch_args, function)(*value)
    assert dragonLauncher.launch_args._launch_args[flag] == result


NOT_SET = object()


@pytest.mark.parametrize("nodes", (NOT_SET, 20, 40))
@pytest.mark.parametrize("tasks_per_node", (NOT_SET, 1, 20))
def test_formatting_launch_args_into_request(
    mock_echo_executable, nodes, tasks_per_node
):
    args = DragonArgBuilder({})
    if nodes is not NOT_SET:
        args.set_nodes(nodes)
    if tasks_per_node is not NOT_SET:
        args.set_tasks_per_node(tasks_per_node)
    req = _as_run_request_view(args, mock_echo_executable, {})

    args = {
        k: v
        for k, v in {
            "nodes": nodes,
            "tasks_per_node": tasks_per_node,
        }.items()
        if v is not NOT_SET
    }
    expected = DragonRunRequestView(
        exe="echo", exe_args=["hello", "world"], path="/tmp", env={}, **args
    )
    assert req.nodes == expected.nodes
    assert req.tasks_per_node == expected.tasks_per_node
    assert req.hostlist == expected.hostlist
    assert req.pmi_enabled == expected.pmi_enabled
