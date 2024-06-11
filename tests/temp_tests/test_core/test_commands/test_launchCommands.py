from smartsim._core.commands.command import Command
from smartsim._core.commands.commandList import CommandList
from smartsim._core.commands.launchCommands import LaunchCommands
from smartsim.settings.launchCommand import LauncherType

pre_cmd = Command(launcher=LauncherType.Slurm, command=["pre", "cmd"])
launch_cmd = Command(launcher=LauncherType.Slurm, command=["launch", "cmd"])
post_cmd = Command(launcher=LauncherType.Slurm, command=["post", "cmd"])
pre_commands_list = CommandList(commands=[pre_cmd])
launch_command_list = CommandList(commands=[launch_cmd])
post_command_list = CommandList(commands=[post_cmd])


def test_launchCommand_init():
    launch_cmd = LaunchCommands(
        prelaunch_commands=pre_commands_list,
        launch_commands=launch_command_list,
        postlaunch_commands=post_command_list,
    )
    assert launch_cmd.prelaunch_command == pre_commands_list
    assert launch_cmd.launch_command == launch_command_list
    assert launch_cmd.postlaunch_command == post_command_list
