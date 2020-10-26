

class StepInfo:

    def __init__(self, status="", returncode=None, output=None, error=None):
        self.status = status
        self.returncode = returncode
        self.output = output
        self.error = error