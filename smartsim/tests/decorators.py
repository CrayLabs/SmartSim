
import pytest

from os import path, mkdir
from decorator import decorator
from shutil import rmtree, which, copyfile


def slurm_controller_test(test_function):
    """Decorator for controller tests that should run on Slurm."""

    experiment_dir = "./controller_test/"

    def wrapper(*args, **kwargs):
        # see if we are on slurm machine
        if not which("srun"):
            pytest.skip()

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)
        mkdir(experiment_dir)
        copyfile('./test_configs/sleep.py',
                 experiment_dir + "sleep.py")
        copyfile('./test_configs/bad.py',
                 experiment_dir + "bad.py")

        test_function()

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        return

    return decorator(wrapper, test_function)


def generator_test(test_function):
    """Decorator for generator tests"""

    experiment_dir = "./generator_test/"

    def wrapper(*args, **kwargs):

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        test_function()

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        return

    return decorator(wrapper, test_function)
