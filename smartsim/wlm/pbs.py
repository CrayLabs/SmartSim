# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import json
import os
import typing as t
from shutil import which

from smartsim.error.errors import LauncherError, SmartSimError

from .._core.launcher.pbs.pbsCommands import qstat


def get_hosts() -> t.List[str]:
    """Get the name of the hosts used in a PBS allocation.

    :returns: Names of the host nodes
    :rtype: list[str]
    :raises SmartSimError: ``PBS_NODEFILE`` is not set
    """
    hosts = []
    if "PBS_NODEFILE" in os.environ:
        node_file_path = os.environ["PBS_NODEFILE"]
        with open(node_file_path, "r", encoding="utf-8") as node_file:
            for line in node_file.readlines():
                host = line.split(".")[0].strip()
                hosts.append(host)
        # account for repeats in PBS_NODEFILE
        return sorted(list(set(hosts)))
    raise SmartSimError(
        "Could not parse interactive allocation nodes from PBS_NODEFILE"
    )


def get_queue() -> str:
    """Get the name of queue in a PBS allocation.

    :returns: The name of the queue
    :rtype: str
    :raises SmartSimError: ``PBS_QUEUE`` is not set
    """
    if "PBS_QUEUE" in os.environ:
        return os.environ["PBS_QUEUE"]
    raise SmartSimError("Could not parse queue from PBS_QUEUE")


def get_tasks() -> int:
    """Get the number of processes on each chunk in a PBS allocation.

    .. note::

        This method requires ``qstat`` be installed on the
        node from which it is run.

    :returns: Then number of tasks in the allocation
    :rtype: int
    :raises LauncherError: Could not access ``qstat``
    :raises SmartSimError: ``PBS_JOBID`` is not set
    """
    if "PBS_JOBID" in os.environ:
        if not which("qstat"):
            raise LauncherError(
                "Attempted PBS function without access to PBS(qstat) at the call site"
            )

        if job_id := os.environ.get("PBS_JOBID"):
            job_info_str, _ = qstat(["-f", "-F", "json", job_id])
            job_info = json.loads(job_info_str)
            return int(job_info["Jobs"][job_id]["resources_used"]["ncpus"])
    raise SmartSimError(
        "Could not parse number of requested tasks without an allocation"
    )


def get_tasks_per_node() -> t.Dict[str, int]:
    """Get the number of processes on each chunk in a PBS allocation.

    .. note::

        This method requires ``qstat`` be installed on the
        node from which it is run.

    :returns: Map of chunks to number of processes on that chunck
    :rtype: dict[str, int]
    :raises LauncherError: Could not access ``qstat``
    :raises SmartSimError: ``PBS_JOBID`` is not set
    """
    if "PBS_JOBID" in os.environ:
        if not which("qstat"):
            raise LauncherError(
                (
                    "Attempted PBS function without access to "
                    "PBS(qstat) at the call site"
                )
            )

        if job_id := os.environ.get("PBS_JOBID"):
            job_info_str, _ = qstat(["-f", "-F", "json", job_id])
            job_info = json.loads(job_info_str)
            chunks_and_ncpus = job_info["Jobs"][job_id]["exec_vnode"]  # type: str

            chunk_cpu_map = {}
            for cunck_and_ncpu in chunks_and_ncpus.split("+"):
                chunk, ncpu = cunck_and_ncpu.strip("()").split(":")
                ncpu = ncpu.lstrip("ncpus=")
                chunk_cpu_map[chunk] = int(ncpu)

            return chunk_cpu_map
    raise SmartSimError("Could not parse tasks per node without an allocation")
