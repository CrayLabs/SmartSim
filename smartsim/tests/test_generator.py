import pytest
from os import path, environ, getcwd
from shutil import rmtree

from smartsim import Experiment
from smartsim.error import SmartSimError, SSConfigError
from smartsim.generation import Generator
from smartsim.tests.decorators import generator_test


test_path = path.join(getcwd(),  "./generator_test/")

exp = Experiment("test")
gen = Generator()

@generator_test
def test_ensemble():
    """Test generating an ensemble of models with the all_perm strategy which
       is the default strategy
    """
    params = {
        "THERMO": [10, 20, 30],
        "STEPS": [10, 20, 30]
        }
    ensemble = exp.create_ensemble("test", params=params)
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(
        test_path,
        ensembles=ensemble
    )

    assert(path.isdir("./generator_test/test/"))
    for i in range(9):
        assert(path.isdir("./generator_test/test/test_" + str(i)))

@generator_test
def test_ensemble_random():
    """test generation of an ensemble using the random generation strategy"""

    gen.set_strategy("random")
    n_models = 10 # number of models to generate

    import numpy as np
    steps = np.random.randint(10, 20, size=(50))
    thermo = np.random.randint(20, 200, size=(50))

    param_dict = {"STEPS": list(steps), "THERMO": list(thermo)}
    ensemble = exp.create_ensemble("random", params=param_dict)
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")

    gen.generate_experiment(
        test_path,
        ensembles=ensemble,
        n_models=10
    )

    assert(path.isdir("./generator_test/random/"))
    for i in range(n_models):
        assert(path.isdir("./generator_test/random/random_" + str(i)))

@generator_test
def test_ensemble_stepped():
    """test the generation of an ensemble using the stepped strategy"""

    gen.set_strategy("step")

    params = {
        "THERMO": [10, 20, 30],
        "STEPS": [10, 20, 30]
        }
    ensemble = exp.create_ensemble("step", params=params)
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(
        test_path,
        ensembles=ensemble
    )

    assert(path.isdir("./generator_test/step/"))
    for i in range(3):
        assert(path.isdir("./generator_test/step/step_" + str(i)))


@generator_test
def test_user_strategy():
    """Test the generation of ensemble using a user given strategy"""

    def step_values(param_names, param_values):
        permutations = []
        for p in zip(*param_values):
            permutations.append(dict(zip(param_names, p)))
        return permutations

    gen.set_strategy(step_values)

    params = {
        "THERMO": [10, 20, 30],
        "STEPS": [10, 20, 30]
        }
    ensemble = exp.create_ensemble("user", params=params)
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(
        test_path,
        ensembles=ensemble
    )
    assert(path.isdir("./generator_test/user/"))
    for i in range(3):
        assert(path.isdir("./generator_test/user/user_" + str(i)))


@generator_test
def test_full_exp():
    """test the generation of all other possible entities within SmartSim
        - orchestrator
        - node with node_files
    """

    node = exp.create_node("node")
    node.attach_generator_files(to_copy="./test_configs/sleep.py")
    orc = exp.create_orchestrator()
    gen.generate_experiment(
        test_path,
        nodes=node,
        orchestrator=orc
    )

    assert(path.isdir("./generator_test/orchestrator"))
    assert(path.isdir("./generator_test/node"))
    assert(path.isfile("./generator_test/node/sleep.py"))


@generator_test
def test_dir_files():
    """test the generate of models and nodes with model and node files that
       are directories with subdirectories and files
    """
    gen.set_strategy("all_perm")

    params = {
        "THERMO": [10, 20, 30],
        "STEPS": [10, 20, 30]
        }
    ensemble = exp.create_ensemble("dir_test", params=params)
    ensemble.attach_generator_files(to_copy="./test_configs/test_dir/")
    node = exp.create_node("node_1")
    node.attach_generator_files(to_copy="./test_configs/test_dir/")

    gen.generate_experiment(
        test_path,
        ensembles=ensemble,
        nodes=node
    )

    assert(path.isdir("./generator_test/dir_test/"))
    for i in range(9):
        model_path = "./generator_test/dir_test/dir_test_" + str(i)
        assert(path.isdir(model_path))
        assert(path.isdir(path.join(model_path, "test_dir_1")))
        assert(path.isfile(path.join(model_path, "test.py")))

    assert(path.isdir("./generator_test/node_1"))
    assert(path.isdir("./generator_test/node_1/test_dir_1"))
    assert(path.isfile("./generator_test/node_1/test.py"))


@generator_test
def test_full_path():
    """Test when a full path is given as a model file"""
    gen.set_strategy("all_perm")

    full_path = path.join(getcwd(), "test_configs/in.atm")

    params = {
        "THERMO": [10, 20, 30],
        "STEPS": [10, 20, 30]
        }
    ensemble = exp.create_ensemble("full_path", params=params)
    ensemble.attach_generator_files(to_configure=full_path)
    gen.generate_experiment(
        test_path,
        ensembles=ensemble
    )

    assert(path.isdir("./generator_test/full_path/"))
    for i in range(9):
        model_path = "./generator_test/full_path/full_path_" + str(i)
        assert(path.isdir(model_path))
        assert(path.isfile(path.join(model_path, "in.atm")))
