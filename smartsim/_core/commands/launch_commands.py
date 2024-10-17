from .command_list import CommandList


class LaunchCommands:
    """Container for aggregating prelaunch commands (e.g. file
    system operations), launch commands, and postlaunch commands
    """

    def __init__(
        self,
        prelaunch_commands: CommandList,
        launch_commands: CommandList,
        postlaunch_commands: CommandList,
    ) -> None:
        """LaunchCommand constructor"""
        self._prelaunch_commands = prelaunch_commands
        self._launch_commands = launch_commands
        self._postlaunch_commands = postlaunch_commands

    @property
    def prelaunch_command(self) -> CommandList:
        """Get the prelaunch command list.
        Return a reference to the command list.
        """
        return self._prelaunch_commands

    @property
    def launch_command(self) -> CommandList:
        """Get the launch command list.
        Return a reference to the command list.
        """
        return self._launch_commands

    @property
    def postlaunch_command(self) -> CommandList:
        """Get the postlaunch command list.
        Return a reference to the command list.
        """
        return self._postlaunch_commands

    def __str__(self) -> str:  # pragma: no cover
        string = "\n\nPrelaunch Command List:\n"
        for pre_cmd in self.prelaunch_command:
            string += f"{pre_cmd}\n"
        string += "\n\nLaunch Command List:\n"
        for launch_cmd in self.launch_command:
            string += f"{launch_cmd}\n"
        string += "\n\nPostlaunch Command List:\n"
        for post_cmd in self.postlaunch_command:
            string += f"{post_cmd}\n"
        return string
