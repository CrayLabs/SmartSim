from enum import Enum

class SchedulerType(Enum):
    """ Schedulers that are supported by
    SmartSim.
    """
    SlurmScheduler = "sbatch"
    PbsScheduler = "qsub"
    LsfScheduler = "bsub"