class SmartSimError(Exception):
    """Base SmartSim error"""

    def __init__(self, message):
        self.msg = message

    def __str__(self):
        return self.msg


class SSUnsupportedError(Exception):
    """raised in the event that a called method isn't supported by SmartSim yet"""

    def __init__(self, message):
        super().__init__(message)


class SSConfigError(SmartSimError):
    """Raised when there is an error in the configuration of SmartSim"""

    def __init__(self, message):
        super().__init__(message)


class SmartSimConnectionError(SmartSimError):
    """Raised when a connection fails between SmartSim entities and the orchestrator"""

    def __init__(self, message):
        super().__init__(message)


class UserStrategyError(SmartSimError):
    """Raised when there is an error in user ensemble generation"""

    def __init__(self, perm_strat):
        message = self.create_message(perm_strat)
        super().__init__(message)

    def create_message(self, perm_strat):
        prefix = "User provided ensemble generation strategy"
        message = "failed to generate valid parameter names and values"
        return " ".join((prefix, str(perm_strat), message))


class ParameterWriterError(SmartSimError):
    def __init__(self, file_path, read=True):
        message = self.create_message(file_path, read)
        super().__init__(message)

    def create_message(self, fp, read):
        if read:
            msg = f"Failed to read configuration file to write at {fp}"
        else:
            msg = f"Failed to write configuration file to {fp}"
        return msg


class EntityExistsError(SmartSimError):
    """Raised when a user tries to create an entity or files/directories for
    an entity and either the entity/files/directories already exist"""

    def __init__(self, message):
        super().__init__(message)


# ------ Launcher Errors ---------


class LauncherError(SmartSimError):
    """Raised when there is an error in the launcher"""

    def __init__(self, message):
        super().__init__(message)


class ShellError(LauncherError):
    """Raised when error arises from function within launcher.shell
    Closely related to error from subprocess(Popen) commands"""

    def __init__(self, message):
        super().__init__(message)


class CommandServerError(LauncherError):
    """Raised when there is a error when communicating with the command server"""

    def __init__(self, message):
        super().__init__(message)


class ClusterLauncherError(LauncherError):
    """Raised when there is an error in the launcher"""

    def __init__(self, message):
        super().__init__(message)


# ------ MPO Errors -----------------


class MPOError(Exception):
    """Raised when errors arise with crayai integration"""

    def __init__(self, message):
        super().__init__(message)