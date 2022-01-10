from shutil import which
from subprocess import run


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
    return "local"
