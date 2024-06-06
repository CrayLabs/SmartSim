from smartsim.settings import LaunchSettings
from smartsim.settings.translators.launch.dragon import DragonArgBuilder
from smartsim.settings.launchCommand import LauncherType
import pytest

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),"2","nodes",id="set_nodes"),
        pytest.param("set_tasks_per_node", (2,),"2","tasks-per-node",id="set_tasks_per_node"),
    ],
)
def test_dragon_class_methods(function, value, flag, result):
    dragonLauncher = LaunchSettings(launcher=LauncherType.Dragon)
    assert isinstance(dragonLauncher._arg_builder,DragonArgBuilder)
    getattr(dragonLauncher.launch_args, function)(*value)
    assert dragonLauncher.launch_args._launch_args[flag] == result