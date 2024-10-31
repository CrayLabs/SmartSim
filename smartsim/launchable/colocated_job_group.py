# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import typing as t
from copy import deepcopy

from .base_job import BaseJob
from .base_job_group import BaseJobGroup

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
