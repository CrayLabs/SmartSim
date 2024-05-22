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
        pass
    
    def insert(self, idx: int, value: BaseJob) -> None:
        if 0 <= idx <= len(self.jobs):
            self.jobs.insert(idx, value)
        else:
            print(f"Invalid index {idx}. Cannot insert value: {value}.")

    def __iter__(self) -> t.Iterator[BaseJob]:
        return iter(self.jobs)
    
    def __getitem__(self, idx: int) -> BaseJob:
        return self.jobs[idx]

    def __setitem__(self, idx: int, value: BaseJob) -> None:
        self.jobs[idx] = value

    def __delitem__(self, idx: int) -> None:
        del self.jobs[idx]

    def __len__(self) -> int:
        return len(self.jobs)

    def __str__(self):  # pragma: no-cover
        string = ""
        string += f"Jobs: {self.jobs}"