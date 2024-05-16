from enum import Enum

class SchedulerType(Enum):
    SlurmLauncher = "sbatch"
    PbsLauncher = "qsub"
    LsfLauncher = "bsub"