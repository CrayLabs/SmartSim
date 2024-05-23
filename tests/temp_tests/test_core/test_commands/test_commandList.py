from smartsim._core.commands.commandList import CommandList
from smartsim._core.commands.command import Command
from smartsim.settingshold.launchCommand import LauncherType

salloc_cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
srun_cmd = Command(launcher=LauncherType.SlurmLauncher, command=["srun", "-n", "1"])
sacct_cmd = Command(launcher=LauncherType.SlurmLauncher, command=["sacct", "--user"])

def test_command_init():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    assert cmd_list.commands == [salloc_cmd,srun_cmd]

def test_command_getitem():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    get_value = cmd_list[0]
    assert get_value == salloc_cmd

def test_command_setitem():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    cmd_list[0] = sacct_cmd
    assert cmd_list.commands == [sacct_cmd,srun_cmd]

def test_command_delitem():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    del(cmd_list.commands[0])
    assert cmd_list.commands == [srun_cmd]

def test_command_len():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    assert len(cmd_list) is 2

def test_command_insert():
    cmd_list = CommandList(commands=[salloc_cmd,srun_cmd])
    cmd_list.insert(0, sacct_cmd)
    assert cmd_list.commands == [sacct_cmd,salloc_cmd,srun_cmd]