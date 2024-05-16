from enum import Enum

class SchedulerType(Enum):
    """ Schedulers that are supported by
    SmartSim.
    """
    SlurmLauncher = "sbatch"
    PbsLauncher = "qsub"
    LsfLauncher = "bsub"