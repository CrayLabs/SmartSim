from enum import Enum

class LauncherType(Enum):
    """ Launchers that are supported by
    SmartSim.
    """
    DragonLauncher = "dragon"
    SlurmLauncher = "slurm"
    PalsLauncher = "pals"
    AlpsLauncher = "aprun"
    LocalLauncher = "local"
    MpiexecLauncher = "mpiexec"
    MpirunLauncher = "mpirun"
    OrterunLauncher = "orterun"
    LsfLauncher = "jsrun"