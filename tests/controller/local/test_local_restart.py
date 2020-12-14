import time
from os import getcwd, path

import pytest

from smartsim import Experiment, constants
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
    O1 = exp.create_orchestrator(path=test_path, port=6780)

    # start all entities for the first time
    local_ctrl.start(M1, M2, O1)
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert(constants.STATUS_FAILED not in statuses)

    local_ctrl.stop_entity(M1)
    local_ctrl.stop_entity(M2)
    local_ctrl.stop_entity_list(O1)

    # model statuses should not be changed to cancelled as they should
    # finish before stop is called
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    assert(all([stat == constants.STATUS_COMPLETED for stat in model_statuses]))
    assert(all([stat == constants.STATUS_CANCELLED for stat in orc_status]))

    # restart all entities
    local_ctrl.start(M1, M2, O1)
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    statuses = orc_status + model_statuses
    assert(constants.STATUS_FAILED not in statuses)

    # TODO: add job history check here
    local_ctrl.stop_entity(M1)
    local_ctrl.stop_entity(M2)
    local_ctrl.stop_entity_list(O1)

    # ensure they all become cancelled jobs
    model_statuses = [local_ctrl.get_entity_status(m) for m in [M1, M2]]
    orc_status = local_ctrl.get_entity_list_status(O1)
    assert(all([stat == constants.STATUS_COMPLETED for stat in model_statuses]))
    assert(all([stat == constants.STATUS_CANCELLED for stat in orc_status]))
