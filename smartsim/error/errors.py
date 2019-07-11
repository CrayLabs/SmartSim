
class SmartSimError(Exception):

    def __init__(self, stage, message):
        self.stg = stage
        self.msg = message

    def __str__(self):
        return "SmartSim Error: (" + self.stg + ") " + self.msg


class SSConfigError(SmartSimError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


class SSUnsupportedError(SmartSimError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


