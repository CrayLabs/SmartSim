import pytest
from .....smartsim._core.command import Command
from smartsim.settingshold.launchCommand import LauncherType

def test_command_initialization_with_valid_inputs():
    cmd = Command(launcher=LauncherType.SlurmLauncher, command=["salloc", "-N", "1"])
    assert cmd.command == ["salloc", "-N", "1"]
    assert cmd.launcher == LauncherType.SlurmLauncher

# def test_command_initialization_with_empty_command_list():
    
# def test_command_initialization_with_invalid_launcher():
    