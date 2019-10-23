
from smartsim import Generator, State
from os import path, environ
from shutil import rmtree

# will error out if not set
SS_HOME = environ['SMARTSIMHOME']

def test_gen_duplicate_configs():
    """Test for the creation of the experiment directory structure"""

    # clean up previous run/test
    experiment_dir = path.join(SS_HOME, "lammps_atm")
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(config="/LAMMPS/simulation.toml")

    # init generator
    GEN = Generator(STATE)
    GEN.generate()

    # assert that experiment direcory was created
    assert(path.isdir(experiment_dir))

    target_1 = path.join(experiment_dir, "atm")
    target_2 = path.join(experiment_dir, "atm-2")
    assert(path.isdir(target_1))
    assert(path.isdir(target_2))

    target_1_model_1 = path.join(target_1, "atm_0")
    target_1_model_2 = path.join(target_1, "atm_1")
    target_2_model_1 = path.join(target_2, "atm-2_0")
    target_2_model_2 = path.join(target_2, "atm-2_1")

    model_dirs = [target_1_model_1, target_1_model_2,
                  target_2_model_1, target_2_model_2]
    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    # clean up this run/test
    experiment_dir = path.join(SS_HOME, "lammps_atm")
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)


def test_gen_with_user_created_models():
    """Test for the creation of the experiment directory structure when both create_target
    and create_model are used (programmatic interface)."""

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25]}
    STATE.create_model("add_1", "atm", {"25": 125})
    STATE.create_target("atm", params=param_dict)
    STATE.create_model("add_2", "atm", {"25": 90})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)

    # init generator
    GEN = Generator(STATE)
    GEN.generate()

    # assert that experiment direcory was created
    assert(path.isdir(experiment_dir))

    target = path.join(experiment_dir, "atm")
    assert(path.isdir(target))

    target_model_1 = path.join(target, "atm_0")
    target_model_2 = path.join(target, "atm_1")
    target_model_3 = path.join(target, "add_1")
    target_model_4 = path.join(target, "add_2")
    

    model_dirs = [target_model_1, target_model_2,
                  target_model_3, target_model_4]
    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

