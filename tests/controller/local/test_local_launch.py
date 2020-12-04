import pytest
import time
from os import path, getcwd
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test_local
from smartsim import constants
# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./controller_test/")
local_ctrl = Controller()
local_ctrl.init_launcher("local")

# --- Tests  -----------------------------------------------

# experiment with non-clustered orchestrator
exp = Experiment("test")

run_settings = {
    "executable": "python",
    "exe_args": "sleep.py --time 10"
}

@controller_test_local
def test_models():
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)

    local_ctrl.start(M1, M2)
    statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    assert(all([stat == constants.STATUS_COMPLETED for stat in statuses]))

@controller_test_local
def test_orchestrator():
    O1 = exp.create_orchestrator(path=test_path, port=6780)

    local_ctrl.start(O1, block=False)
    statuses = local_ctrl.get_entity_list_status(O1)
    assert(constants.STATUS_FAILED not in statuses)
    local_ctrl.stop_entity_list(O1)
    statuses = local_ctrl.get_entity_list_status(O1)
    assert(all([stat == constants.STATUS_CANCELLED for stat in statuses]))


@controller_test_local
def test_ensemble():

    ensemble = exp.create_ensemble("e1")
    M3 = exp.create_model("m3", path=test_path, run_settings=run_settings)
    ensemble.add_model(M3)

    local_ctrl.start(ensemble, block=False)
    statuses = local_ctrl.get_entity_list_status(ensemble)
    assert(all([stat in constants.LIVE_STATUSES for stat in statuses]))
    local_ctrl.stop_entity_list(ensemble)
    statuses = local_ctrl.get_entity_list_status(ensemble)
    assert(all([stat == constants.STATUS_CANCELLED for stat in statuses]))