
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
    STATE = State(config="LAMMPS/simulation.toml")

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


