
from smartsim import Generator, State
from os import path, environ
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError

# will error out if not set
SS_HOME = environ['SMARTSIMHOME']

def test_state_api_with_create_target_create_model():
    """Test for the creation of the experiment directory structure when both create_target
    and create_model are used (programmatic interface)."""

    EXPERIMENT = "lammps_atm"
    # create a state with the LAMMPS configuration file
    STATE = State(experiment=EXPERIMENT)

    # Create a 'default_target', and an 'atm' target.
    # Each target should have 1 model after this, because we have manually added
    # a model to each target; if we then fed this state into a generator, the
    # 'atm' state would have _two_ models.
    param_dict = {"25": [20, 25]}
    model_1_params = {"25": 90}
    model_2_params = {"25": 125}
    STATE.create_target("atm", params=param_dict)
    STATE.create_model("add_1", "atm", model_1_params)
    STATE.create_model("add_1", params=model_2_params)

    # assert that we have created 2 targets.
    assert(len(STATE.targets) == 2)

    # current code assumes targets are appended, but we should confirm.
    atm_target = 0
    default_target = 1
    if STATE.targets[0].name == "default_target":
        atm_target = 1
        default_target = 0

    # assert that each target has one existing model.
    assert(len(STATE.targets[default_target]._models) == 1)
    assert(len(STATE.targets[atm_target]._models) == 1)

    # assert that we have correctly copied our params into our 'atm' target.
    assert(STATE.targets[atm_target].params == param_dict)
    # assert that our params are an empty dictionary for our default target.
    assert(STATE.targets[default_target].params == {})

    # assert that our model params are set appropriately.
    assert(STATE.targets[atm_target]._models["add_1"].param_dict == model_1_params)
    assert(STATE.targets[default_target]._models["add_1"].param_dict == model_2_params)