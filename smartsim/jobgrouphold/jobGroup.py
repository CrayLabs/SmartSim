import typing as t
from copy import deepcopy
from .baseJobGroup import BaseJobGroup, BaseJob
class Job: pass # assume Job class exists until Julias implementation

class JobGroup(BaseJobGroup):
    """A job group holds references to multiple jobs that
    will be executed all at the same time when resources
    permit.  Execution is blocked until resources are available.
    """
    def __init__(
        self,
        jobs: t.List[BaseJob],
    ) -> None:
        super().__init__()
        self._jobs = jobs

    @property
    def jobs(self) -> t.List[BaseJob]:
         return self._jobs
    
    def __str__(self):  # pragma: no-cover
        string = ""
        string += f"Jobs: {self.jobs}"