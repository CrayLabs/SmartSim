import typing as t
from abc import abstractmethod
from collections.abc import MutableSequence
from copy import deepcopy

from smartsim.launchable.launchable import Launchable

from .basejob import BaseJob


class BaseJobGroup(Launchable, MutableSequence):
    """Highest level ABC of a group of jobs that can be
    launched
    """

    def __init__(self) -> None:
        super().__init__()

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

    def __getitem__(self, idx: int) -> BaseJob:
        """Retrieves the job at the specified index (idx)."""
        return self.jobs[idx]

    def __setitem__(self, idx: int, value: BaseJob) -> None:
        """Sets the job at the specified index (idx) to the given value."""
        self.jobs[idx] = deepcopy(value)

    def __delitem__(self, idx: int) -> None:
        """Deletes the job at the specified index (idx)."""
        del self.jobs[idx]

    def __len__(self) -> int:
        """Returns the total number of jobs in the collection."""
        return len(self.jobs)

    def __str__(self):  # pragma: no-cover
        """Returns a string representation of the collection of jobs."""
        string = ""
        string += f"Jobs: {self.jobs}"
