import typing as t
from .baseJobGroup import BaseJobGroup, BaseJob
from copy import deepcopy

class ColocatedJobGroup(BaseJobGroup):
    """A colocated job group holds references to multiple jobs that
    will be executed all at the same time when resources
    permit. Execution is blocked until resources are available.
    """
    def __init__(
        self,
        jobs: t.List[BaseJob],
    ) -> None:
        super().__init__()
        self._jobs = deepcopy(jobs)

    @property
    def jobs(self) -> t.List[BaseJob]:
        """This property method returns a list of BaseJob objects.
        It represents the collection of jobs associated with an
        instance of the BaseJobGroup abstract class.
        """
        return self._jobs
    
    def __str__(self):  # pragma: no-cover
        """Returns a string representation of the collection of
        colocated job groups.
        """
        string = ""
        string += f"Colocated Jobs: {self.jobs}"