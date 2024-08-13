from __future__ import annotations

import typing as t
from copy import deepcopy

from smartsim.log import get_logger

from .._core.utils.helpers import check_name
from .basejob import BaseJob
from .baseJobGroup import BaseJobGroup

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from typing_extensions import Self


@t.final
class JobGroup(BaseJobGroup):
    """A job group holds references to multiple jobs that
    will be executed all at the same time when resources
    permit. Execution is blocked until resources are available.
    """

    def __init__(
        self,
        jobs: t.List[BaseJob],
        name: str = "job_group",
    ) -> None:
        super().__init__()
        self._jobs = deepcopy(jobs)
        self._name = name
        check_name(self._name)

    @property
    def name(self) -> str:
        """Retrieves the name of the JobGroup."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the JobGroup."""
        check_name(name)
        logger.info(f'Overwriting Job name from "{self._name}" to "{name}"')
        self._name = name

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
        job groups.
        """
        return f"Job Groups: {self.jobs}"
