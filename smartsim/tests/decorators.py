
import pytest

from os import path, mkdir
from decorator import decorator
from shutil import rmtree, which, copyfile
from subprocess import Popen, PIPE
from smartsim.utils import get_logger
logger = get_logger()


def controller_test(test_function):
    """Decorator for controller tests"""

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

def orchestrator_test_slurm(test_function):

    exp_base_path = "./orchestrator_test"
    exp_1_dir = "./orchestrator_test/exp_1/"
    exp_2_dir = "./orchestrator_test/exp_2/"

    def wrapper(*args, **kwargs):
        if not which("srun"):
            pytest.skip()

        if path.isdir(exp_base_path):
            rmtree(exp_base_path)
        mkdir(exp_base_path)
        mkdir(exp_1_dir)
        mkdir(exp_2_dir)

        copyfile('./test_configs/reconnect_sim.py',
                 exp_1_dir + "reconnect_sim.py")
        copyfile('./test_configs/reconnect_node.py',
                 exp_2_dir + "reconnect_node.py")

        test_function()

        if path.isdir(exp_base_path):
            rmtree(exp_base_path)

        return

    return decorator(wrapper, test_function)

def orchestrator_test_local(test_function):

    exp_base_path = "./orchestrator_test"
    exp_1_dir = "./orchestrator_test/exp_1/"
    exp_2_dir = "./orchestrator_test/exp_2/"

    def wrapper(*args, **kwargs):

        if path.isdir(exp_base_path):
            rmtree(exp_base_path)
        mkdir(exp_base_path)
        mkdir(exp_1_dir)
        mkdir(exp_2_dir)

        copyfile('./test_configs/reconnect_sim.py',
                 exp_1_dir + "reconnect_sim.py")
        copyfile('./test_configs/reconnect_node.py',
                 exp_2_dir + "reconnect_node.py")

        test_function()

        if path.isdir(exp_base_path):
            rmtree(exp_base_path)

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


def modelwriter_test(test_function):
    """Decorator for ModelWriter tests"""

    experiment_dir = "./modelwriter_test/"

    def wrapper(*args, **kwargs):

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)
        mkdir(experiment_dir)

        test_function()

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        return

    return decorator(wrapper, test_function)


def python_client_test(test_dir):
    """Decorator for Python client"""

    experiment_dir = "./clients/python/python_client_test/"

    def wrapper(test_function, *args, **kwargs):

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)
        mkdir(experiment_dir)

        copyfile(test_dir + 'node.py',
                experiment_dir + "node.py")
        copyfile(test_dir + 'simulation.py',
                experiment_dir + "simulation.py")

        test_function()

        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        return decorator(wrapper, test_function)
    return decorator(wrapper)
