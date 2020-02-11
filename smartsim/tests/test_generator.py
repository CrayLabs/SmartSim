from smartsim import Generator, State
from os import path, environ
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError
import pytest

from smartsim.utils.log import get_logger
logger = get_logger(__name__)



def test_generator_basic():
    """Test for the creation of the experiment directory structure when both create_ensemble
    and create_model are used."""

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    state = State(experiment="lammps_atm")

    param_dict = {"STEPS": [20, 25]}
    state.create_ensemble("atm", params=param_dict)
    state.create_model("add_1", "atm", {"STEPS": 90})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "../../examples/LAMMPS/in.atm"
    gen = Generator(state, model_files=base_config)
    gen.generate()

    # assert that experiment directory was created
    assert(path.isdir(experiment_dir))
    ensemble = path.join(experiment_dir, "atm")
    assert(path.isdir(ensemble))

    ensemble_model_1 = path.join(ensemble, "atm_0")
    ensemble_model_2 = path.join(ensemble, "atm_1")
    ensemble_model_3 = path.join(ensemble, "add_1")

    model_dirs = [ensemble_model_1, ensemble_model_2,
                  ensemble_model_3]
    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)


def test_model_exists():
    """Test error thrown if same model is created twice."""

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    STATE = State(experiment="lammps_atm")
    STATE.create_model("add_1", params={"STEPS": 50})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "../../examples/LAMMPS/in.atm"
    gen = Generator(STATE, model_files=base_config)
    gen.generate()

    try:
        STATE.create_model("add_1", params={"STEPS": 90})
        raise SmartSimError("Model name: add_1 has been incorrectly replaced")
    except SSModelExistsError:
        logger.info("Model exists error correctly thrown")
        pass

    assert(path.isdir(experiment_dir))
    ensemble_1 = path.join(experiment_dir, "default")
    ensemble_model_1 = path.join(ensemble_1, "add_1")

    # check for model dir and listed configuration file
    assert(path.isdir(ensemble_model_1))
    assert(path.isfile(path.join(ensemble_model_1, "in.atm")))

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_select_strategy_user_function():
    """A test of the generator using a user supplied function.
    """

    def raise_error(param_names, param_values):
        raise NotImplementedError

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    STATE = State(experiment="lammps_atm")
    param_dict = {"STEPS": [20, 25]}
    STATE.create_ensemble("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "../../examples/LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    GEN.set_strategy(raise_error)
    strategy_failed_out = False
    try:
        GEN.generate()
    except NotImplementedError:
        strategy_failed_out = True

    assert(strategy_failed_out)

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)


def test_gen_select_strategy_default():

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    STATE = State(experiment="lammps_atm")

    param_dict = {"STEPS": [20, 25], "THERMO": [10]}
    STATE.create_ensemble("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "../../examples/LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    GEN.set_strategy("all_perm")
    GEN.generate()
    assert(len(STATE.ensembles[0]) == 2)

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)


def test_gen_random_strategy():

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    STATE = State(experiment="lammps_atm")

    # make some parameter values
    import numpy as np
    steps = np.random.randint(10, 20, size=(50))
    thermo = np.random.randint(20, 200, size=(50))

    param_dict = {"STEPS": list(steps), "THERMO": list(thermo)}
    STATE.create_ensemble("atm", params=param_dict)

    base_config = "../../examples/LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    GEN.set_strategy("random")
    GEN.generate(n_models=10)

    print(STATE)
    assert(len(STATE.ensembles[0]) == 10)

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)


def test_gen_step_strategy():

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    STATE = State(experiment="lammps_atm")
    param_dict = {"STEPS": [20, 25, 30], "THERMO": [10, 20, 30]}
    STATE.create_ensemble("atm", params=param_dict)

    base_config = "../../examples/LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    GEN.set_strategy("step")
    GEN.generate()
    assert(len(STATE.ensembles[0]) == 3)

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_generator_no_model_files():
    """Test for the creation of the experiment directory structure when both create_ensemble
    and create_model are used but without specification of model files."""

    experiment_dir = "./lammps_atm"
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    state = State(experiment="lammps_atm")

    state.create_ensemble("atm")
    state.create_ensemble("atm-2")
    state.create_model("add_1", "atm", {})

    # Supply the generator with state and build directories
    gen = Generator(state)
    gen.generate()

    # assert that experiment directory was created
    assert(path.isdir(experiment_dir))
    ensemble_1 = path.join(experiment_dir, "atm")
    assert(path.isdir(ensemble_1))
    ensemble_2 = path.join(experiment_dir, "atm-2")
    assert(path.isdir(ensemble_2))

    # assert model subdirectory was created
    ensemble_model_1 = path.join(ensemble_1, "add_1")
    assert(path.isdir(ensemble_model_1))

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)