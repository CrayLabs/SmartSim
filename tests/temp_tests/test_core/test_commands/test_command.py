from smartsim._core.commands.command import Command
from smartsim.settingshold.launchCommand import LauncherType

def test_command_init():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
    assert cmd.command == ["salloc", "-N", "1"]
    assert cmd.launcher == LauncherType.SlurmLauncher

def test_command_getitem():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
    get_value = cmd[0]
    assert get_value == "salloc"

def test_command_setitem():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
    cmd[0] = "srun"
    cmd[1] = "-n"
    assert cmd.command == ["srun", "-n", "1"]

def test_command_delitem():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1", "--constraint", "P100"])
    del(cmd.command[3])
    del(cmd.command[3])
    assert cmd.command == ["salloc", "-N", "1"]

def test_command_len():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
    assert len(cmd) is 3

def test_command_insert():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["-N", "1"])
    cmd.insert(0, "salloc")
    assert cmd.command == ["salloc", "-N", "1"]