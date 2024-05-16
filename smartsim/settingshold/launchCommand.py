from enum import Enum

class LauncherType(Enum):
    DragonLauncher = "dragon"
    SlurmLauncher = "slurm"
    PalsLauncher = "pals"
    AlpsLauncher = "aprun"
    LocalLauncher = "local"
    MpiexecLauncher = "mpiexec"
    MpirunLauncher = "mpirun"
    OrterunLauncher = "orterun"
    LsfLauncher = "jsrun"