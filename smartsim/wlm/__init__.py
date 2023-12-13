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

import os
from shutil import which
from subprocess import run
import typing as t

from ..error import SSUnsupportedError
from . import pbs as _pbs
from . import slurm as _slurm


def detect_launcher() -> str:
    """Detect available launcher."""
    # Precedence: PBS, Cobalt, LSF, Slurm, local
    if which("qsub") and which("qstat") and which("qdel"):
        qsub_version = run(
            ["qsub", "--version"],
            shell=False,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        if "pbs" in (qsub_version.stdout).lower():
            return "pbs"
        if "cobalt" in (qsub_version.stdout).lower():
            return "cobalt"
    if all(
        [which("bsub"), which("jsrun"), which("jslist"), which("bjobs"), which("bkill")]
    ):
        return "lsf"
    if all(
        [
            which("sacct"),
            which("srun"),
            which("salloc"),
            which("sbatch"),
            which("scancel"),
            which("sstat"),
            which("sinfo"),
        ]
    ):
        return "slurm"
    # Systems like ThetaGPU don't have
    # Cobalt or PBS on compute nodes
    if "COBALT_JOBID" in os.environ:
        return "cobalt"
    if "PBS_JOBID" in os.environ:
        return "pbs"
    return "local"


def get_hosts(launcher: t.Optional[str] = None) -> t.List[str]:
    """Get the name of the hosts used in an allocation.

    :param launcher: Name of the WLM to use to collect allocation info. If no launcher
                     is provided ``detect_launcher`` is used to select a launcher.
    :type launcher: str | None
    :returns: Names of the hosts
    :rtype: list[str]
    :raises SSUnsupportedError: User attempted to use an unsupported WLM
    """
    if launcher is None:
        launcher = detect_launcher()
    if launcher == "pbs":
        return _pbs.get_hosts()
    if launcher == "slurm":
        return _slurm.get_hosts()
    raise SSUnsupportedError(f"SmartSim cannot get hosts for launcher `{launcher}`")


def get_queue(launcher: t.Optional[str] = None) -> str:
    """Get the name of the queue used in an allocation.

    :param launcher: Name of the WLM to use to collect allocation info. If no launcher
                     is provided ``detect_launcher`` is used to select a launcher.
    :type launcher: str | None
    :returns: Name of the queue
    :rtype: str
    :raises SSUnsupportedError: User attempted to use an unsupported WLM
    """
    if launcher is None:
        launcher = detect_launcher()
    if launcher == "pbs":
        return _pbs.get_queue()
    if launcher == "slurm":
        return _slurm.get_queue()
    raise SSUnsupportedError(f"SmartSim cannot get queue for launcher `{launcher}`")


def get_tasks(launcher: t.Optional[str] = None) -> int:
    """Get the number of tasks in an allocation.

    :param launcher: Name of the WLM to use to collect allocation info. If no launcher
                     is provided ``detect_launcher`` is used to select a launcher.
    :type launcher: str | None
    :returns: Number of tasks
    :rtype: int
    :raises SSUnsupportedError: User attempted to use an unsupported WLM
    """
    if launcher is None:
        launcher = detect_launcher()
    if launcher == "pbs":
        return _pbs.get_tasks()
    if launcher == "slurm":
        return _slurm.get_tasks()
    raise SSUnsupportedError(f"SmartSim cannot get tasks for launcher `{launcher}`")


def get_tasks_per_node(launcher: t.Optional[str] = None) -> t.Dict[str, int]:
    """Get a map of nodes in an allocation to the number of tasks on each node.

    :param launcher: Name of the WLM to use to collect allocation info. If no launcher
                     is provided ``detect_launcher`` is used to select a launcher.
    :type launcher: str | None
    :returns: Map of nodes to number of processes on that node
    :rtype: dict[str, int]
    :raises SSUnsupportedError: User attempted to use an unsupported WLM
    """
    if launcher is None:
        launcher = detect_launcher()
    if launcher == "pbs":
        return _pbs.get_tasks_per_node()
    if launcher == "slurm":
        return _slurm.get_tasks_per_node()
    raise SSUnsupportedError(
        f"SmartSim cannot get tasks per node for launcher `{launcher}`"
    )
