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
from .arguments import BatchArguments
from .arguments.batch.lsf import BsubBatchArguments
from .arguments.batch.pbs import QsubBatchArguments
from .arguments.batch.slurm import SlurmBatchArguments
from .baseSettings import BaseSettings
from .batchCommand import SchedulerType
from .common import StringArgument

logger = get_logger(__name__)


class BatchSettings(BaseSettings):
    """The BatchSettings class manages the configuration and execution of batch jobs
    across the resources of an HPC system.

    BatchSettings is designed to be extended by a BatchArguments child class that
    corresponds to the scheduler provided during initialization. The supported schedulers
    are Slurm, PBS, and LSF. Using the BatchSettings class, users can:

    - Set the scheduler type of a batch job.
    - Configure batch arguments and environment variables.
    - Access and modify custom batch arguments.
    - Update environment variables.
    - Retrieve information associated with the ``BatchSettings`` object.
        - The scheduler value (BatchSettings.scheduler).
        - The derived BatchArguments child class (BatchSettings.scheduler_args).
        - The set environment variables (BatchSettings.env_vars).
        - A formatted output of set batch arguments (BatchSettings.format_batch_args).
    """

    def __init__(
        self,
        batch_scheduler: t.Union[SchedulerType, str],
        scheduler_args: StringArgument | None = None,
        env_vars: StringArgument | None = None,
    ) -> None:
        """Initialize a BatchSettings instance.

        Example of initializing BatchSettings:

        .. highlight:: python
        .. code-block:: python

            sbatch_settings = BatchSettings(batch_scheduler="slurm")
            # OR
            sbatch_settings = BatchSettings(batch_scheduler=SchedulerType.Slurm)

        The "batch_scheduler" of SmartSim BatchSettings will determine the
        child type assigned to the BatchSettings.scheduler_args attribute.
        The example above will return a SlurmBatchArguments object. Using
        the object, users may access the child class functions to set batch
        configurations. For example:

        .. highlight:: python
        .. code-block:: python

            sbatch_settings.scheduler_args.set_nodes(5)
            sbatch_settings.scheduler_args.set_cpus_per_task(2)

        To set customized batch arguments, use the set() function provided by
        the BatchSettings child class. For example:

        .. highlight:: python
        .. code-block:: python

            sbatch_settings.scheduler_args.set(key="nodes", value="6")

        If the key already exists in the existing batch arguments, the value will
        be overwritten.

        :param batch_scheduler: The type of scheduler to initialize (e.g., Slurm, PBS, LSF)
        :param scheduler_args: A dictionary of arguments for the scheduler, where the keys
            are strings and the values can be either strings or None. This argument is optional
            and defaults to None.
        :param env_vars: Environment variables for the batch settings, where the keys
            are strings and the values can be either strings or None. This argument is
            also optional and defaults to None.
        :raises ValueError: Raises if the batch_scheduler provided does not exist.
        """
        try:
            self._batch_scheduler = SchedulerType(batch_scheduler)
        except ValueError:
            raise ValueError(f"Invalid scheduler type: {batch_scheduler}") from None
        self._arguments = self._get_arguments(scheduler_args)
        self.env_vars = env_vars or {}

    @property
    def scheduler(self) -> str:
        """Return the scheduler type."""
        return self._batch_scheduler.value

    @property
    def scheduler_args(self) -> BatchArguments:
        """Return the BatchArguments child class."""
        return self._arguments

    @property
    def env_vars(self) -> StringArgument:
        """Return an immutable list of attached environment variables."""
        return copy.deepcopy(self._env_vars)

    @env_vars.setter
    def env_vars(self, value: t.Dict[str, str | None]) -> None:
        """Set the environment variables."""
        self._env_vars = copy.deepcopy(value)

    def _get_arguments(self, scheduler_args: StringArgument | None) -> BatchArguments:
        """Map the Scheduler to the BatchArguments. This method should only be
        called once during construction.

        :param scheduler_args: A mapping of arguments names to values to be
            used to initialize the arguments
        :returns: The appropriate type for the settings instance.
        """
        if self._batch_scheduler == SchedulerType.Slurm:
            return SlurmBatchArguments(scheduler_args)
        elif self._batch_scheduler == SchedulerType.Lsf:
            return BsubBatchArguments(scheduler_args)
        elif self._batch_scheduler == SchedulerType.Pbs:
            return QsubBatchArguments(scheduler_args)
        else:
            raise ValueError(f"Invalid scheduler type: {self._batch_scheduler}")

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments to preview

        :return: formatted batch arguments
        """
        return self._arguments.format_batch_args()

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nScheduler: {self.scheduler}{self.scheduler_args}"
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string
