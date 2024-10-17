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

import abc
import collections.abc
import typing as t
import uuid

from typing_extensions import Self

from smartsim.status import JobStatus
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from smartsim.experiment import Experiment

_T_contra = t.TypeVar("_T_contra", contravariant=True)


def create_job_id() -> LaunchedJobID:
    return LaunchedJobID(str(uuid.uuid4()))


class LauncherProtocol(collections.abc.Hashable, t.Protocol[_T_contra]):
    """The protocol defining a launcher that can be used by a SmartSim
    experiment
    """

    @classmethod
    @abc.abstractmethod
    def create(cls, exp: Experiment, /) -> Self:
        """Create an new launcher instance from and to be used by the passed in
        experiment instance

        :param: An experiment to use the newly created launcher instance
        :returns: The newly constructed launcher instance
        """

    @abc.abstractmethod
    def start(self, launchable: _T_contra, /) -> LaunchedJobID:
        """Given input that this launcher understands, create a new process and
        issue a launched job id to query the status of the job in future.

        :param launchable: The input to start a new process
        :returns: The id to query the status of the process in future
        """

    @abc.abstractmethod
    def get_status(
        self, *launched_ids: LaunchedJobID
    ) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Given a collection of launched job ids, return a mapping of id to
        current status of the launched job. If a job id is no recognized by the
        launcher, a `smartsim.error.errors.LauncherJobNotFound` error should be
        raised.

        :param launched_ids: The collection of ids of launched jobs to query
            for current status
        :raises smartsim.error.errors.LauncherJobNotFound: If at least one of
            the ids of the `launched_ids` collection is not recognized.
        :returns: A mapping of launched id to current status
        """

    @abc.abstractmethod
    def stop_jobs(
        self, *launched_ids: LaunchedJobID
    ) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Given a collection of launched job ids, cancel the launched jobs

        :param launched_ids: The ids of the jobs to stop
        :raises smartsim.error.errors.LauncherJobNotFound: If at least one of
            the ids of the `launched_ids` collection is not recognized.
        :returns: A mapping of launched id to status upon cancellation
        """
