
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

class SSModelExistsError(SmartSimError):
    """An error that is raised when a duplicate model (or model with the same name)
    is added to a ensemble."""
    def __init__(self, message):
        super().__init__(message)

class SmartSimConnectionError(SmartSimError):
    """Raised when a connection fails between SmartSim entities and the orchestrator"""
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
