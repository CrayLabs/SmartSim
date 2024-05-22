import typing as t
from abc import abstractmethod
from copy import deepcopy
class BaseJob: pass # assume BaseJob class exists until Julias implementation
class Launchable: pass # assume BaseJob class exists until Julias implementation
class MutableSequence: pass

class BaseJobGroup(Launchable, MutableSequence):
    """Highest level ABC of a group of jobs that can be 
    launched
    """
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def jobs(self) -> t.List[BaseJob]:
        return self.jobs

    def __getitem___(self, idx) -> BaseJob:
        return self.jobs[idx]

    def __setitem__(self, idx, value) -> None:
        self.jobs[idx] = deepcopy(value)

    def __delitem__(self, idx) -> None:
        del self.jobs[idx]
# after that just need to do dunder string
    def __len__(self) -> int:
        return len(self.jobs)

    def __str__(self):  # pragma: no-cover
        string = ""
        string += f"Jobs: {self.jobs}"