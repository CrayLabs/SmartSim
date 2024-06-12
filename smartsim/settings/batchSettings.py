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

import copy
import typing as t

from smartsim.log import get_logger

from .._core.utils.helpers import fmt_dict
from .baseSettings import BaseSettings
from .batchCommand import SchedulerType
from .builders import BatchArgBuilder
from .builders.batch.lsf import BsubBatchArgBuilder
from .builders.batch.pbs import QsubBatchArgBuilder
from .builders.batch.slurm import SlurmBatchArgBuilder
from .common import StringArgument

logger = get_logger(__name__)


class BatchSettings(BaseSettings):
    def __init__(
        self,
        batch_scheduler: t.Union[SchedulerType, str],
        scheduler_args: t.Dict[str, t.Union[str, None]] | None = None,
        env_vars: StringArgument | None = None,
    ) -> None:
        try:
            self._batch_scheduler = SchedulerType(batch_scheduler)
        except ValueError:
            raise ValueError(f"Invalid scheduler type: {batch_scheduler}") from None
        self._arg_builder = self._get_arg_builder(scheduler_args)
        self.env_vars = env_vars or {}

    @property
    def scheduler(self) -> str:
        """Return the launcher name."""
        return self._batch_scheduler.value

    @property
    def batch_scheduler(self) -> str:
        """Return the scheduler name."""
        return self._batch_scheduler.value

    @property
    def scheduler_args(self) -> BatchArgBuilder:
        """Return the batch argument translator."""
        return self._arg_builder

    @property
    def env_vars(self) -> StringArgument:
        """Return an immutable list of attached environment variables."""
        return copy.deepcopy(self._env_vars)

    @env_vars.setter
    def env_vars(self, value: t.Dict[str, str | None]) -> None:
        """Set the environment variables."""
        self._env_vars = copy.deepcopy(value)

    def _get_arg_builder(
        self, scheduler_args: StringArgument | None
    ) -> BatchArgBuilder:
        """Map the Scheduler to the BatchArgBuilder"""
        if self._batch_scheduler == SchedulerType.Slurm:
            return SlurmBatchArgBuilder(scheduler_args)
        elif self._batch_scheduler == SchedulerType.Lsf:
            return BsubBatchArgBuilder(scheduler_args)
        elif self._batch_scheduler == SchedulerType.Pbs:
            return QsubBatchArgBuilder(scheduler_args)
        else:
            raise ValueError(f"Invalid scheduler type: {self._batch_scheduler}")

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        return self._arg_builder.format_batch_args()

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nScheduler: {self.scheduler}"
        if self.scheduler_args:
            string += str(self.scheduler_args)
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string
