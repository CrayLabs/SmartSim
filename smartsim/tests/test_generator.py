from smartsim import Generator, State
from os import path, environ
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError

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

    # assert that experiment directory was created
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

def test_gen_with_create_target_create_model():
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
    STATE.create_target("atm", params=param_dict)
    STATE.create_model("add_1", "atm", {"25": 90})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    GEN.generate()

    # assert that experiment directory was created
    assert(path.isdir(experiment_dir))

    target = path.join(experiment_dir, "atm")
    assert(path.isdir(target))

    target_model_1 = path.join(target, "atm_0")
    target_model_2 = path.join(target, "atm_1")
    target_model_3 = path.join(target, "add_1")

    model_dirs = [target_model_1, target_model_2,
                  target_model_3]
    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_with_user_created_models():
    """Test for the creation of the experiment directory structure when only
    create_model is used (programmatic interface); we should create a new, empty target."""

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # We should be able to create 3 new targets.
    STATE.create_model("add_1", "atm_1", {"25": 10})
    STATE.create_model("add_2", "atm_1", {"25": 20})

    STATE.create_model("add_1", "atm_2", {"25": 30})
    STATE.create_model("add_2", "atm_2", {"25": 40})

    STATE.create_model("add_1", params={"25": 50})
    STATE.create_model("add_2", params={"25": 60})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)

    # init generator
    GEN.generate()

    # assert that experiment directory was created
    assert(path.isdir(experiment_dir))

    target_1 = path.join(experiment_dir, "atm_1")
    target_2 = path.join(experiment_dir, "atm_2")
    target_3 = path.join(experiment_dir, "default_target")
    assert(path.isdir(target_1))
    assert(path.isdir(target_2))
    assert(path.isdir(target_3))

    target_model_1 = path.join(target_1, "add_1")
    target_model_2 = path.join(target_1, "add_2")
    target_model_3 = path.join(target_2, "add_1")
    target_model_4 = path.join(target_2, "add_2")
    target_model_5 = path.join(target_3, "add_1")
    target_model_6 = path.join(target_3, "add_2")

    model_dirs = [target_model_1, target_model_2,
                  target_model_3, target_model_4,
                  target_model_5, target_model_6]
    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_overwrite_create_model():
    """Test for the creation of the experiment directory structure when only
    create_model is used (programmatic interface); we should create a new, empty target."""

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # We should be able to create 2 new targets.
    STATE.create_model("add_1", "atm_1", {"25": 10})
    STATE.create_model("add_1", params={"25": 50})

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)

    # init generator
    GEN.generate()

    # attempt to replace an existing model.  We should error out when we try to add it.
    # if we don't, THAT'S when we want to error out.
    try:
        STATE.create_model("add_1", params={"25": 90})
        raise SmartSimError("Generator testing",
                            "Model name: add_1 has been incorrectly replaced")
    except SSModelExistsError:
        pass

    try:
        STATE.create_model("add_1", "atm_1", params={"25": 90})
        raise SmartSimError("Generator testing",
                            "Model name: add_1 has been incorrectly replaced")
    except SSModelExistsError:
        pass

    # assert that experiment directory was created
    assert(path.isdir(experiment_dir))

    target_1 = path.join(experiment_dir, "atm_1")
    target_2 = path.join(experiment_dir, "default_target")
    assert(path.isdir(target_1))
    assert(path.isdir(target_2))

    target_model_1 = path.join(target_1, "add_1")
    target_model_2 = path.join(target_2, "add_1")

    model_dirs = [target_model_1, target_model_2]

    # check for model dir and listed configuration file
    for model in model_dirs:
        assert(path.isdir(model))
        assert(path.isfile(path.join(model, "in.atm")))

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_select_strategy_user_function():
    """A test of the generator using a user supplied function.
    """

    def raise_error(param_names, param_values):
        raise NotImplementedError

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25]}
    STATE.create_target("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    #GEN.set_strategy(create_all_permutations)
    GEN.set_strategy(raise_error)
    strategy_failed_out = False
    try:
        GEN.generate()
    except NotImplementedError:
        #  We should have successfully failed out.
        strategy_failed_out = True

    assert(strategy_failed_out)

    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_select_strategy_user_string():

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25]}
    STATE.create_target("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    #GEN.set_strategy(create_all_permutations)
    GEN.set_strategy("generation_strategies.raise_error")
    strategy_failed_out = False
    try:
        GEN.generate()
    except NotImplementedError:
        #  We should have successfully failed out.
        strategy_failed_out = True

    assert(strategy_failed_out)
    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_select_strategy_default():

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25], "5": [10]}
    STATE.create_target("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    #GEN.set_strategy(create_all_permutations)
    GEN.set_strategy("all_perm")
    GEN.generate()
    assert(len(STATE.targets[0].get_models()) == 2)

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_random_strategy():

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25], "5": [10]}
    STATE.create_target("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    #GEN.set_strategy(create_all_permutations)
    GEN.set_strategy("random")
    GEN.generate(n_models=10)
    assert(len(STATE.targets[0].get_models()) == 2)

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

def test_gen_step_strategy():

    # clean up previous run/test
    EXPERIMENT = "lammps_atm"
    experiment_dir = path.join(SS_HOME, EXPERIMENT)
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)

    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    param_dict = {"25": [20, 25, 30], "5": [10, 20, 30]}
    STATE.create_target("atm", params=param_dict)

    # Supply the generator with necessary files to run the simulation
    # and generate the specified models
    base_config = "LAMMPS/in.atm"
    GEN = Generator(STATE, model_files=base_config)
    #GEN.set_strategy(create_all_permutations)
    GEN.set_strategy("step")
    GEN.generate()
    assert(len(STATE.targets[0].get_models()) == 3)

    # clean up this run/test
    if path.isdir(experiment_dir):
        rmtree(experiment_dir)
