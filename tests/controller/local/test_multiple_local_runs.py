
import pytest
import time
from os import path, getcwd
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test_local
# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./controller_test/")

@controller_test_local
def test_multiple_runs():
    """test calling start multiple times in a row but not with
       the same objects (aka not a restart)"""

    # initialize experiment entities
    exp = Experiment("Multiple-Run-Tests", launcher="local")
    ctrl = exp._control
    run_settings = {
        "executable": "python",
        "exe_args": "sleep.py"
    }
    M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
    M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
    O1 = exp.create_orchestrator(path=test_path)

    ctrl.start(M1, M2)
    ctrl.poll(3, False, True)
    statuses = [ctrl.get_entity_status(m) for m in [M1, M2]]
    assert("failed" not in statuses)

    ctrl.start(O1)
    time.sleep(5)
    statuses = ctrl.get_entity_list_status(O1)
    assert(all([x == "running" for x in statuses]))
    ctrl.stop_entity_list(O1)