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
from abc import ABC, abstractmethod

from smartsim.log import get_logger

from ..._core.utils.helpers import fmt_dict

logger = get_logger(__name__)


class BatchArguments(ABC):
    """Abstract base class that defines all generic scheduler
    argument methods that are not supported.  It is the
    responsibility of child classes for each launcher to translate
    the input parameter to a properly formatted launcher argument.
    """

    def __init__(self, batch_args: t.Dict[str, str | None] | None) -> None:
        self._batch_args = copy.deepcopy(batch_args) or {}
        """A dictionary of batch arguments"""

    @abstractmethod
    def scheduler_str(self) -> str:
        """Get the string representation of the launcher"""
        pass

    @abstractmethod
    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param account: account id
        """
        pass

    @abstractmethod
    def set_queue(self, queue: str) -> None:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        pass

    @abstractmethod
    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        :param walltime: wall time
        """
        pass

    @abstractmethod
    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        """
        pass

    @abstractmethod
    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        pass

    @abstractmethod
    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        pass

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nScheduler Arguments:\n{fmt_dict(self._batch_args)}"
        return string
