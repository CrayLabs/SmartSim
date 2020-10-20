import pytest
from os import path, environ, getcwd
from shutil import rmtree
import json

from smartsim import Experiment
from smartsim.error import SmartSimError, SSConfigError
from smartsim.generation import Generator
from smartsim.utils.test.decorators import generator_test


test_path = path.join(getcwd(), "./generator_test/")
exp = Experiment("test")
gen = Generator(test_path)

gen_run_settings = {"executable": "python"}


@generator_test
def test_ensemble():
    """Test generating an ensemble of models with the all_perm strategy which
    is the default strategy
    """
    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=gen_run_settings)
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(ensemble)

    assert(len(ensemble) == 9)
    assert path.isdir("./generator_test/test/")
    for i in range(9):
        assert path.isdir("./generator_test/test/test_" + str(i))


@generator_test
def test_ensemble_random():
    """test generation of an ensemble using the random generation strategy"""

    n_models = 10  # number of models to generate

    import numpy as np

    steps = np.random.randint(10, 20, size=(50))
    thermo = np.random.randint(20, 200, size=(50))

    param_dict = {"STEPS": list(steps), "THERMO": list(thermo)}
    ensemble = exp.create_ensemble(
        "random",
        params=param_dict,
        perm_strategy="random",
        run_settings=gen_run_settings,
        n_models=10
    )
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(ensemble)

    assert(len(ensemble) == 10)
    assert path.isdir("./generator_test/random/")
    for i in range(n_models):
        assert path.isdir("./generator_test/random/random_" + str(i))


@generator_test
def test_ensemble_stepped():
    """test the generation of an ensemble using the stepped strategy"""

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble(
        "step", params=params, perm_strategy="step", run_settings=gen_run_settings
    )
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(ensemble)

    assert(len(ensemble) == 3)
    assert path.isdir("./generator_test/step/")
    for i in range(3):
        assert path.isdir("./generator_test/step/step_" + str(i))


@generator_test
def test_user_strategy():
    """Test the generation of ensemble using a user given strategy"""

    def step_values(param_names, param_values):
        permutations = []
        for p in zip(*param_values):
            permutations.append(dict(zip(param_names, p)))
        return permutations

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble(
        "user", params=params, perm_strategy=step_values, run_settings=gen_run_settings
    )
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(ensemble)

    assert(len(ensemble) == 3)
    assert path.isdir("./generator_test/user/")
    for i in range(3):
        assert path.isdir("./generator_test/user/user_" + str(i))


@generator_test
def test_full_exp():
    """test the generation of all other possible entities within SmartSim
    - orchestrator
    - node with node_files
    """

    model = exp.create_model("model", run_settings=gen_run_settings)
    model.attach_generator_files(to_copy="./test_configs/sleep.py")
    orc = exp.create_orchestrator()
    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble(
        "test_ens", params=params, run_settings=gen_run_settings
    )
    ensemble.attach_generator_files(to_configure="./test_configs/in.atm")
    gen.generate_experiment(ensemble, orc, model)

    # test for ensemble
    assert path.isdir("./generator_test/test_ens/")
    for i in range(9):
        assert path.isdir("./generator_test/test_ens/test_ens_" + str(i))

    # test for orc dir
    assert path.isdir("./generator_test/database")

    # test for model file
    assert path.isdir("./generator_test/model")
    assert path.isfile("./generator_test/model/sleep.py")


@generator_test
def test_dir_files():
    """test the generate of models with files that
    are directories with subdirectories and files
    """

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble(
        "dir_test", params=params, run_settings=gen_run_settings
    )
    ensemble.attach_generator_files(to_copy="./test_configs/test_dir/")

    gen.generate_experiment(ensemble)

    assert path.isdir("./generator_test/dir_test/")
    for i in range(9):
        model_path = "./generator_test/dir_test/dir_test_" + str(i)
        assert path.isdir(model_path)
        assert path.isdir(path.join(model_path, "test_dir_1"))
        assert path.isfile(path.join(model_path, "test.py"))


@generator_test
def test_full_path():
    """Test when a full path is given as a model file"""

    full_path = path.join(getcwd(), "test_configs/in.atm")

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble(
        "full_path", params=params, run_settings=gen_run_settings
    )
    ensemble.attach_generator_files(to_configure=full_path)
    gen.generate_experiment(ensemble)

    assert path.isdir("./generator_test/full_path/")
    for i in range(9):
        model_path = "./generator_test/full_path/full_path_" + str(i)
        assert path.isdir(model_path)
        assert path.isfile(path.join(model_path, "in.atm"))


@generator_test
def test_model_generation():

    params = {"placeholder_1": 30}
    model = exp.create_model("model", params=params, run_settings=gen_run_settings)
    model.attach_generator_files(to_configure="./test_configs/sample.json")
    gen.generate_experiment(model)

    # test for model files
    assert path.isdir("./generator_test/model")
    assert path.isfile("./generator_test/model/sample.json")

    with open("./generator_test/model/sample.json", "r") as read_file:
        config = json.load(read_file)

    assert(config["param1"] == 30)
