
from smartsim import Generator, State
from os import path, environ
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError

# will error out if not set
SS_HOME = environ['SMARTSIMHOME']

def test_state_api_create_target():
    """Test to ensure create_target works as expected"""

    EXPERIMENT = "lammps_atm"
    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # Create an 'atm' target.  The target should exist and have 0 models.
    param_dict = {"25": [20, 25]}
    STATE.create_target("atm", params=param_dict)

    # assert that we have created 1 target.
    assert(len(STATE.targets) == 1)

    # assert that our target doesn't have any models.
    assert(len(STATE.targets[0].get_models()) == 0)

    # assert that out params are correctly set.
    assert(STATE.targets[0].get_target_params() == param_dict)

    # assert that our experiment name and path is properly set.
    assert(STATE.targets[0].get_target_dir() == path.join(SS_HOME, EXPERIMENT, "atm"))

def test_state_api_create_model():
    """Test to ensure create_model works as expected (and generates 
    targets appropriately)"""

    EXPERIMENT = "lammps_atm"
    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # Create a 'default_target', and an 'atm' target.
    # Each target should have 1 model after this, because we have manually added
    # a model to each target; if we then fed this state into a generator, the
    # 'atm' state would have _two_ models.
    params = []
    params.append({"25": 90})
    params.append({"25": 125})
    STATE.create_model("add_1", "atm", params[0])
    STATE.create_model("add_1", params=params[1])
    names = ["atm", "default_target"]

    # assert that we have created 2 targets.
    assert(len(STATE.targets) == 2)

    # assert that each target has one existing model.
    assert(len(STATE.targets[0].get_models()) == 1)
    assert(len(STATE.targets[1].get_models()) == 1)

    # assert that our model and target params are set appropriately.
    for target in range(0, 2):
        # our target params should be empty.
        assert(STATE.targets[target].get_target_params() == {})
        assert(STATE.targets[target].get_target_dir() == path.join(SS_HOME, EXPERIMENT, names[target]))
        for key in params[target].keys():
            assert(STATE.targets[target].get_models()["add_1"].get_param_value(key) == params[target][key])


def test_state_api_create_target_with_create_model():
    """Test to ensure using create model with create target works 
    without conflicting."""

    EXPERIMENT = "lammps_atm"
    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # Create a 'default_target', and an 'atm' target.
    # Each target should have 1 model after this, because we have manually added
    # a model to each target; if we then fed this state into a generator, the
    # 'atm' state would have _two_ models.
    params = []
    params.append([{"25": [20, 25]}, {"25": 90}])
    params.append([{}, {"25": 125}])
    STATE.create_target("atm", params=params[0][0])
    STATE.create_model("add_1", "atm", params[0][1])
    STATE.create_model("add_1", params=params[1][1])
    names = ["atm", "default_target"]

    # assert that we have created 2 targets.
    assert(len(STATE.targets) == 2)

    # assert that each target has one existing model.
    assert(len(STATE.targets[0].get_models()) == 1)
    assert(len(STATE.targets[1].get_models()) == 1)

    # assert that our model and target params are set appropriately.
    for target in range(0, 2):
        # our target params should be empty.
        assert(STATE.targets[target].get_target_params() == params[target][0])
        assert(STATE.targets[target].get_target_dir() == path.join(SS_HOME, EXPERIMENT, names[target]))
        for key in params[target][1].keys():
            assert(STATE.targets[target].get_models()["add_1"].get_param_value(key) == params[target][1][key])