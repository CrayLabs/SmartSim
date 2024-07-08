from __future__ import annotations

import typing as t
from copy import deepcopy

from .basejob import BaseJob
from .baseJobGroup import BaseJobGroup

if t.TYPE_CHECKING:
    from typing_extensions import Self


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

    @t.overload
    def __getitem__(self, idx: int) -> BaseJob: ...
    @t.overload
    def __getitem__(self, idx: slice) -> Self: ...
    def __getitem__(self, idx: int | slice) -> BaseJob | Self:
        """Retrieves the job at the specified index (idx)."""
        jobs = self.jobs[idx]
        if isinstance(jobs, BaseJob):
            return jobs
        return type(self)(jobs)

    def __str__(self) -> str:  # pragma: no-cover
        """Returns a string representation of the collection of
        colocated job groups.
        """
        return f"Colocated Jobs: {self.jobs}"
