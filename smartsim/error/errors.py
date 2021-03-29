class SmartSimError(Exception):
    """Base SmartSim error"""


class SSUnsupportedError(Exception):
    """raised in the event that a called method isn't supported by SmartSim yet"""


class SSConfigError(SmartSimError):
    """Raised when there is an error in the configuration of SmartSim"""


class EntityExistsError(SmartSimError):
    """Raised when a user tries to create an entity or files/directories for
    an entity and either the entity/files/directories already exist"""


class UserStrategyError(SmartSimError):
    """Raised when there is an error with model creation inside an ensemble
    that is from a user provided permutation strategy"""

    def __init__(self, perm_strat):
        message = self.create_message(perm_strat)
        super().__init__(message)

    def create_message(self, perm_strat):
        prefix = "User provided ensemble generation strategy"
        message = "failed to generate valid parameter names and values"
        return " ".join((prefix, str(perm_strat), message))


class ParameterWriterError(SmartSimError):
    """Raised in the event that input parameter files for a model
    could not be written.
    """

    def __init__(self, file_path, read=True):
        message = self.create_message(file_path, read)
        super().__init__(message)

    def create_message(self, fp, read):
        if read:
            msg = f"Failed to read configuration file to write at {fp}"
        else:
            msg = f"Failed to write configuration file to {fp}"
        return msg


# ------ Launcher Errors ---------


class LauncherError(SmartSimError):
    """Raised when there is an error in the launcher"""


class ShellError(LauncherError):
    """Raised when error arises from function within launcher.shell
    Closely related to error from subprocess(Popen) commands"""

    def __init__(self, message, shell_error, command_list):
        msg = self.create_message(message, shell_error, command_list)
        super().__init__(msg)

    def create_message(self, message, shell_error, command_list):
        if isinstance(command_list, list):
            command_list = " ".join(command_list)
        msg = message + "\n"
        msg += f"\nCommand: {command_list}"
        if shell_error:
            msg += f"\nError from shell: {shell_error}"
        return msg
