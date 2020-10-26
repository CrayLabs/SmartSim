import pytest
import time
from os import path, getcwd
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test_local

test_path = path.join(getcwd(),  "./controller_test/")

@controller_test_local
def test_restart():

    # initialize experiment entities
    exp = Experiment("Restart-Test", launcher="local")
    local_ctrl = exp._control
    run_settings = {
        "executable": "python",
        "exe_args": "sleep.py"
    }
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
    O1 = exp.create_orchestrator(path=test_path)

    # start all entities for the first time
    local_ctrl.start(M1, M2, O1)
    local_ctrl.poll(3, False, True)
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert("failed" not in statuses)

    local_ctrl.stop_entity(M1)
    local_ctrl.stop_entity(M2)
    local_ctrl.stop_entity_list(O1)


    # restart all entities
    local_ctrl.start(M1, M2, O1)
    local_ctrl.poll(3, False, True)
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert("failed" not in statuses)

    # TODO: add job history check here
    local_ctrl.stop_entity(M1)
    local_ctrl.stop_entity(M2)
    local_ctrl.stop_entity_list(O1)