import os
from shutil import which
from subprocess import run

from ..error import SSUnsupportedError
from . import pbs, slurm


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


def get_hosts():
    launcher = detect_launcher()
    if launcher == "pbs":
        return pbs.get_hosts()
    if launcher == "slurm":
        return slurm.get_hosts()
    raise SSUnsupportedError(f"SmartSim cannot get hosts for launcher `{launcher}`")


def get_queue():
    launcher = detect_launcher()
    if launcher == "pbs":
        return pbs.get_queue()
    if launcher == "slurm":
        return slurm.get_queue()
    raise SSUnsupportedError(f"SmartSim cannot get queue for launcher `{launcher}`")


def get_tasks():
    launcher = detect_launcher()
    if launcher == "pbs":
        return pbs.get_tasks()
    if launcher == "slurm":
        return slurm.get_tasks()
    raise SSUnsupportedError(f"SmartSim cannot get tasks for launcher `{launcher}`")


def get_tasks_per_node():
    launcher = detect_launcher()
    if launcher == "pbs":
        return pbs.get_tasks_per_node()
    if launcher == "slurm":
        return slurm.get_tasks_per_node()
    raise SSUnsupportedError(
        f"SmartSim cannot get tasks per node for launcher `{launcher}`"
    )
