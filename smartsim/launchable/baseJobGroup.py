from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from copy import deepcopy

from smartsim.launchable.launchable import Launchable

from .basejob import BaseJob


class BaseJobGroup(Launchable, MutableSequence[BaseJob], ABC):
    """Highest level ABC of a group of jobs that can be
    launched
    """

    @property
    @abstractmethod
    def jobs(self) -> t.List[BaseJob]:
        """This property method returns a list of BaseJob objects.
        It represents the collection of jobs associated with an
        instance of the BaseJobGroup abstract class.
        """
        pass

    def insert(self, idx: int, value: BaseJob) -> None:
        """Inserts the given value at the specified index (idx) in
        the list of jobs. If the index is out of bounds, the method
        prints an error message.
        """
        self.jobs.insert(idx, value)

    def __iter__(self) -> t.Iterator[BaseJob]:
        """Allows iteration over the jobs in the collection."""
        return iter(self.jobs)

    @t.overload
    def __setitem__(self, idx: int, value: BaseJob) -> None: ...
    @t.overload
    def __setitem__(self, idx: slice, value: t.Iterable[BaseJob]) -> None: ...
    def __setitem__(
        self, idx: int | slice, value: BaseJob | t.Iterable[BaseJob]
    ) -> None:
        """Sets the job at the specified index (idx) to the given value."""
        if isinstance(idx, int):
            if not isinstance(value, BaseJob):
                raise TypeError("Can only assign a `BaseJob`")
            self.jobs[idx] = deepcopy(value)
        else:
            if not isinstance(value, t.Iterable):
                raise TypeError("Can only assign an iterable")
            self.jobs[idx] = (deepcopy(val) for val in value)

    def __delitem__(self, idx: int | slice) -> None:
        """Deletes the job at the specified index (idx)."""
        del self.jobs[idx]

    def __len__(self) -> int:
        """Returns the total number of jobs in the collection."""
        return len(self.jobs)

    def __str__(self) -> str:  # pragma: no-cover
        """Returns a string representation of the collection of jobs."""
        return f"Jobs: {self.jobs}"
