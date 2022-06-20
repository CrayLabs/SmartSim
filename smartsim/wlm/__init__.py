import os
from shutil import which
from subprocess import run

from ..error import SSUnsupportedError
from . import pbs as _pbs
from . import slurm as _slurm


def detect_launcher():
    """Detect available launcher."""
    # Precedence: PBS, Cobalt, LSF, Slurm, local
    if which("qsub") and which("qstat") and which("qdel"):
        qsub_version = run(
            ["qsub", "--version"], shell=False, capture_output=True, encoding="utf-8"
        )
        if "pbs" in (qsub_version.stdout).lower():
            return "pbs"
        if "cobalt" in (qsub_version.stdout).lower():
            return "cobalt"
    if (
        which("bsub")
        and which("jsrun")
        and which("jslist")
        and which("bjobs")
        and which("bkill")
    ):
        return "lsf"
    if (
        which("sacct")
        and which("srun")
        and which("salloc")
        and which("sbatch")
        and which("scancel")
        and which("sstat")
        and which("sinfo")
    ):
        return "slurm"
    # Systems like ThetaGPU don't have
    # Cobalt or PBS on compute nodes
    if "COBALT_JOBID" in os.environ:
        return "cobalt"
    if "PBS_JOBID" in os.environ:
        return "pbs"
    return "local"


def get_hosts(launcher=None):
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


def get_queue(launcher=None):
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


def get_tasks(launcher=None):
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


def get_tasks_per_node(launcher=None):
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
