
class SmartSimError(Exception):

    def __init__(self, stage, message):
        self.stg = stage
        self.msg = message

    def __str__(self):
        return "(" + self.stg + ") " + self.msg


class SSConfigError(SmartSimError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


class SSUnsupportedError(SmartSimError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


class SSModelExistsError(SmartSimError):
    '''An error that is raised when a duplicate model (or model with the same name)
    is added to a target.'''
    def __init__(self, stage, message):
        super().__init__(stage, message)

