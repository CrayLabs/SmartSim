
class MpoError(Exception):

    def __init__(self, stage, message):
        self.stg = stage
        self.msg = message

    def __str__(self):
        return "MPO Error: (" + self.stg + ") " + self.msg


class MpoConfigError(MpoError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


class MpoUnsupportedError(MpoError):

    def __init__(self, stage, message):
        super().__init__(stage, message)


