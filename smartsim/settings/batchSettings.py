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
import copy

from smartsim.log import get_logger
from .._core.utils.helpers import fmt_dict
from .common import StringArgument
from .batchCommand import SchedulerType
from .translators.batch.pbs import QsubBatchArgTranslator
from .translators.batch.slurm import SlurmBatchArgTranslator
from .translators.batch.lsf import BsubBatchArgTranslator
from .translators import BatchArgTranslator
from .baseSettings import BaseSettings

logger = get_logger(__name__)

class BatchSettings(BaseSettings):
    def __init__(
        self,
        batch_scheduler: t.Union[SchedulerType, str],
        scheduler_args: t.Optional[t.Dict[str, t.Union[str,int,float,None]]] = None,
        env_vars: t.Optional[StringArgument] = None,
    ) -> None:
        try:
            self._batch_scheduler = SchedulerType(batch_scheduler)
        except ValueError:
            raise ValueError(f"Invalid scheduler type: {batch_scheduler}")
        self._arg_translator = self._get_arg_builder(scheduler_args)
        self.env_vars = env_vars or {}

    @property
    def batch_scheduler(self) -> str:
        """Return the scheduler name.
        """
        return self._batch_scheduler.value

    @property
    def scheduler_args(self) -> BatchArgTranslator:
        """Return the batch argument translator.
        """
        # Is a deep copy needed here?
        return self._arg_translator

    @property
    def env_vars(self) -> StringArgument:
        """Return an immutable list of attached environment variables.
        """
        return copy.deepcopy(self._env_vars)

    @env_vars.setter
    def env_vars(self, value: t.Mapping[str, str]) -> None:
        """Set the environment variables.
        """
        self._env_vars = copy.deepcopy(value)

    def _get_arg_builder(self, scheduler_args) -> BatchArgTranslator:
        """ Map the Scheduler to the BatchArgTranslator
        """
        if self._batch_scheduler == SchedulerType.SlurmScheduler:
            return SlurmBatchArgTranslator(scheduler_args)
        elif self._batch_scheduler == SchedulerType.LsfScheduler:
            return BsubBatchArgTranslator(scheduler_args)
        elif self._batch_scheduler == SchedulerType.PbsScheduler:
            return QsubBatchArgTranslator(scheduler_args)
        else:
            raise ValueError(f"Invalid scheduler type: {self._batch_scheduler}")
    
    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        return self._arg_translator.format_batch_args()

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nScheduler: {self.arg_translator.scheduler_str}"
        if self.scheduler_args:
            string += f"\nScheduler Arguments:\n{fmt_dict(self.scheduler_args)}"
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string