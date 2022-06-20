import json
import os
from shutil import which

from smartsim.error.errors import LauncherError, SmartSimError

from .._core.launcher.pbs.pbsCommands import qstat


def get_hosts():
    """Get the name of the hosts used in a PBS allocation.

    :returns: Names of the host nodes
    :rtype: list[str]
    :raises SmartSimError: ``PBS_NODEFILE`` is not set
    """
    hosts = []
    if "PBS_NODEFILE" in os.environ:
        node_file = os.environ["PBS_NODEFILE"]
        with open(node_file, "r") as f:
            for line in f.readlines():
                host = line.split(".")[0].strip()
                hosts.append(host)
        # account for repeats in PBS_NODEFILE
        return sorted(list(set(hosts)))
    raise SmartSimError(
        "Could not parse interactive allocation nodes from PBS_NODEFILE"
    )


def get_queue():
    """Get the name of queue in a PBS allocation.

    :returns: The name of the queue
    :rtype: str
    :raises SmartSimError: ``PBS_QUEUE`` is not set
    """
    if "PBS_QUEUE" in os.environ:
        return os.environ.get("PBS_QUEUE")
    raise SmartSimError("Could not parse queue from SLURM_JOB_PARTITION")


def get_tasks():
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
                (
                    "Attempted PBS function without access to "
                    "PBS(qstat) at the call site"
                )
            )
        job_id = os.environ.get("PBS_JOBID")
        job_info_str, _ = qstat(["-f", "-F", "json", job_id])
        job_info = json.loads(job_info_str)
        return int(job_info["Jobs"][job_id]["resources_used"]["ncpus"])
    raise SmartSimError(
        "Could not parse number of requested tasks without an allocation"
    )


def get_tasks_per_node():
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
        job_id = os.environ.get("PBS_JOBID")
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
