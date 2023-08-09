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
import typing as t
from shutil import which

from .._core.launcher.slurm.slurmCommands import salloc, scancel, scontrol, sinfo
from .._core.launcher.slurm.slurmParser import parse_salloc, parse_salloc_error
from .._core.launcher.util.launcherUtil import ComputeNode, Partition
from .._core.utils.helpers import init_default
from ..error import (
    AllocationError,
    LauncherError,
    SmartSimError,
    SSReservedKeywordError,
)
from ..log import get_logger

logger = get_logger(__name__)


def get_allocation(
    nodes: int = 1,
    time: t.Optional[str] = None,
    account: t.Optional[str] = None,
    options: t.Optional[t.Dict[str, str]] = None,
) -> str:
    """Request an allocation

    This function requests an allocation with the specified arguments.
    Anything passed to the options will be processed as a Slurm
    argument and appended to the salloc command with the appropriate
    prefix (e.g. "-" or "--").

    The options can be used to pass extra settings to the
    workload manager such as the following for Slurm:

        - nodelist="nid00004"

    For arguments without a value, pass None or and empty
    string as the value. For Slurm:

        - exclusive=None

    :param nodes: number of nodes for the allocation, defaults to 1
    :type nodes: int, optional
    :param time: wall time of the allocation, HH:MM:SS format, defaults to None
    :type time: str, optional
    :param account: account id for allocation, defaults to None
    :type account: str, optional
    :param options: additional options for the slurm wlm, defaults to None
    :type options: dict[str, str], optional
    :raises LauncherError: if the allocation is not successful
    :return: the id of the allocation
    :rtype: str
    """
    if not which("salloc"):
        raise LauncherError(
            "Attempted slurm function without access to slurm(salloc) at the call site"
        )

    options = init_default({}, options, dict)

    salloc_args = _get_alloc_cmd(nodes, time, account, options=options)
    debug_msg = " ".join(salloc_args[1:])
    logger.debug(f"Allocation settings: {debug_msg}")

    _, err = salloc(salloc_args)
    alloc_id = parse_salloc(err)
    if alloc_id:
        logger.info(f"Allocation successful with Job ID: {str(alloc_id)}")
    else:
        logger.debug(err)
        error = parse_salloc_error(err)
        if not error:
            logger.error(err)
            raise AllocationError("Error retrieving Slurm allocation")
        raise AllocationError(error)
    return str(alloc_id)


def release_allocation(alloc_id: str) -> None:
    """Free an allocation's resources

    :param alloc_id: allocation id
    :type alloc_id: str
    :raises LauncherError: if allocation could not be freed
    """
    if not which("scancel"):
        raise LauncherError(
            "Attempted slurm function without access to slurm(salloc) at the call site"
        )

    logger.info(f"Releasing allocation: {alloc_id}")
    returncode, _, _ = scancel([str(alloc_id)])

    if returncode != 0:
        logger.error(f"Unable to revoke your allocation for jobid {str(alloc_id)}")
        logger.error(
            "The job may have already timed out, or you may "
            "need to cancel the job manually"
        )
        raise AllocationError(
            f"Unable to revoke your allocation for jobid  {str(alloc_id)}"
        )

    logger.info(f"Successfully freed allocation {alloc_id}")


def validate(nodes: int = 1, ppn: int = 1, partition: t.Optional[str] = None) -> bool:
    """Check that there are sufficient resources in the provided Slurm partitions.

    if no partition is provided, the default partition is found and used.

    :param nodes: Override the default node count to validate, defaults to 1
    :type nodes: int, optional
    :param ppn: Override the default processes per node to validate, defaults to 1
    :type ppn: int, optional
    :param partition: partition to validate, defaults to None
    :type partition: str, optional
    :raises: LauncherError
    :returns: True if resources are available, False otherwise
    :rtype: bool
    """
    sys_partitions = _get_system_partition_info()

    n_avail_nodes = 0
    avail_nodes = set()

    p_name = partition
    if p_name is None or p_name == "default":
        try:
            p_name = get_default_partition()
        except LauncherError as e:
            raise LauncherError(
                "No partition provided and default partition could not be found"
            ) from e

    if not p_name in sys_partitions:
        raise LauncherError(f"Partition {p_name} is not found on this system")

    for node in sys_partitions[p_name].nodes:
        if node.ppn is not None and node.ppn >= ppn:
            avail_nodes.add(node)

    n_avail_nodes = len(avail_nodes)
    logger.debug(f"Found {n_avail_nodes} nodes that match the constraints provided")

    if n_avail_nodes < nodes:
        logger.warning(
            f"{nodes} nodes are not available on the specified partitions.  Only "
            f"{n_avail_nodes} nodes available."
        )
        return False

    logger.info("Successfully validated Slurm with sufficient resources")
    return True


def get_default_partition() -> str:
    """Returns the default partition from Slurm

    This default partition is assumed to be the partition with
    a star following its partition name in sinfo output

    :returns: the name of the default partition
    :rtype: str
    """
    sinfo_output, _ = sinfo(["--noheader", "--format", "%P"])

    default = None
    for line in sinfo_output.split("\n"):
        if line.endswith("*"):
            default = line.strip("*")

    if not default:
        raise LauncherError("Could not find default partition!")
    return default


def _get_system_partition_info() -> t.Dict[str, Partition]:
    """Build a dictionary of slurm partitions
    :returns: dict of Partition objects
    :rtype: dict
    """

    sinfo_output, _ = sinfo(["--noheader", "--format", "%R %n %c"])

    partitions: t.Dict[str, Partition] = {}
    for line in sinfo_output.split("\n"):
        line = line.strip()
        if line == "":
            continue

        p_info = line.split(" ")
        p_name = p_info[0]
        p_node = p_info[1]
        p_ppn = int(p_info[2])

        if not p_name in partitions:
            partitions.update({p_name: Partition()})

        partitions[p_name].name = p_name
        partitions[p_name].nodes.add(ComputeNode(node_name=p_node, node_ppn=p_ppn))

    return partitions


def _get_alloc_cmd(
    nodes: int,
    time: t.Optional[str] = None,
    account: t.Optional[str] = None,
    options: t.Optional[t.Dict[str, str]] = None,
) -> t.List[str]:
    """Return the command to request an allocation from Slurm with
    the class variables as the slurm options."""

    salloc_args = [
        "--no-shell",
        "-N",
        str(nodes),
        "-J",
        "SmartSim",
    ]
    if time:
        salloc_args.extend(["-t", time])
    if account:
        salloc_args.extend(["-A", str(account)])

    arguments = set(options.keys() if options is not None else {})
    invalid = {"t", "time", "N", "nodes", "A", "account"}

    if valid := arguments.intersection(invalid):
        raise SSReservedKeywordError(
            f"Expecting time, nodes, account as an argument. Also received: {valid}"
        )

    for opt, val in (options or {}).items():
        short_arg = bool(len(str(opt)) == 1)
        prefix = "-" if short_arg else "--"
        if not val:
            salloc_args += [prefix + opt]
        else:
            if short_arg:
                salloc_args += [prefix + opt, str(val)]
            else:
                salloc_args += ["=".join((prefix + opt, str(val)))]
    return salloc_args


def get_hosts() -> t.List[str]:
    """Get the name of the nodes used in a slurm allocation.

    .. note::

        This method requires access to ``scontrol`` from the node
        on which it is run

    :returns: Names of the host nodes
    :rtype: list[str]
    :raises LauncherError: Could not access ``scontrol``
    :raises SmartSimError: ``SLURM_JOB_NODELIST`` is not set
    """
    if "SLURM_JOB_NODELIST" in os.environ:
        if not which("scontrol"):
            raise LauncherError(
                (
                    "Attempted slurm function without access to "
                    "slurm(scontrol) at the call site"
                )
            )
        nodelist, _ = scontrol(
            ["show", "hostnames", os.environ.get("SLURM_JOB_NODELIST", "")]
        )
        return sorted(nodelist.split())
    raise SmartSimError("Could not parse allocation nodes from SLURM_JOB_NODELIST")


def get_queue() -> str:
    """Get the name of queue in a slurm allocation.

    :returns: The name of the queue
    :rtype: str
    :raises SmartSimError: ``SLURM_JOB_PARTITION`` is not set
    """
    if job_partition := os.environ.get("SLURM_JOB_PARTITION", None):
        return job_partition
    raise SmartSimError("Could not parse queue from SLURM_JOB_PARTITION")


def get_tasks() -> int:
    """Get the number of tasks in a slurm allocation.

    :returns: Then number of tasks in the allocation
    :rtype: int
    :raises SmartSimError: ``SLURM_NTASKS`` is not set
    """
    if ntasks_str := os.environ.get("SLURM_NTASKS", 0):
        return int(ntasks_str)
    raise SmartSimError("Could not parse number of requested tasks from SLURM_NTASKS")


def get_tasks_per_node() -> t.Dict[str, int]:
    """Get the number of tasks per each node in a slurm allocation.

    .. note::

        This method requires access to ``scontrol`` from the node
        on which it is run

    :returns: Map of nodes to number of tasks on that node
    :rtype: dict[str, int]
    :raises SmartSimError: ``SLURM_TASKS_PER_NODE`` is not set
    """
    if "SLURM_TASKS_PER_NODE" in os.environ:
        tasks_per_node_strs = os.environ.get("SLURM_TASKS_PER_NODE", "").split(",")
        tasks_per_node_list = []
        for tasks_per_node_str in tasks_per_node_strs:
            if "(" in tasks_per_node_str:
                tasks, quantity = tasks_per_node_str.split("(")
                quantity = quantity.rstrip(")").lstrip("x")
                tasks_per_node_list.extend([int(tasks)] * int(quantity))
            else:
                tasks_per_node_list.append(int(tasks_per_node_str))
        return dict(zip(get_hosts(), tasks_per_node_list))
    raise SmartSimError("Could not parse tasks per node from SLURM_TASKS_PER_NODE")
