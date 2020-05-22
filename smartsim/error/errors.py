
class SmartSimError(Exception):
    """Base SmartSim error"""
    def __init__(self, message):
        self.msg = message

    def __str__(self):
        return self.msg

class SSConfigError(SmartSimError):
    """Raised when there is an error in the configuration of SmartSim"""
    def __init__(self, message):
        super().__init__(message)

class SSUnsupportedError(SmartSimError):
    """raised in the event that a called method isn't supported by SmartSim yet"""
    def __init__(self, message):
        super().__init__(message)

class SmartSimConnectionError(SmartSimError):
    """Raised when a connection fails between SmartSim entities and the orchestrator"""
    def __init__(self, message):
        super().__init__(message)


# ------ Generation Errors -------


class GenerationError(SmartSimError):
    """Raised when there is an error within the generator"""
    def __init__(self, message):
        super().__init__(message)

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
