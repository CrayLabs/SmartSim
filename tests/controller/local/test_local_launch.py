import pytest
import time
from os import path, getcwd
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test_local
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
    "exe_args": "sleep.py --time 3"
}

@controller_test_local
def test_models():
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)

    local_ctrl.start(M1, M2)
    local_ctrl.poll(3, False, True)
    statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    assert("failed" not in statuses)

@controller_test_local
def test_orchestrator():
    O1 = exp.create_orchestrator(path=test_path)

    local_ctrl.start(O1)
    time.sleep(5)
    statuses = local_ctrl.get_entity_list_status(O1)
    assert("failed" not in statuses)
    local_ctrl.stop_entity_list(O1)

@controller_test_local
def test_ensemble():

    ensemble = exp.create_ensemble("e1")
    M3 = exp.create_model("m3", path=test_path, run_settings=run_settings)
    ensemble.add_model(M3)

    local_ctrl.start(ensemble)
    time.sleep(5)
    statuses = local_ctrl.get_entity_list_status(ensemble)
    assert("failed" not in statuses)
    local_ctrl.stop_entity_list(ensemble)
