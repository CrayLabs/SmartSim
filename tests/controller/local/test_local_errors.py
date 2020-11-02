import time
import pytest
from os import path, getcwd
from smartsim import Experiment
from smartsim.control import Controller
from smartsim.error import SSUnsupportedError, SmartSimError
from smartsim.utils.test.decorators import controller_test_local
from smartsim import constants

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")

@controller_test_local
def test_failed_status():
    """Test when a failure occurs deep into model execution"""

    exp = Experiment("test_report_failure", launcher="local")
    ctrl = exp._control

    run_settings_report_failure = {
        "executable": "python",
        "exe_args": "bad.py --time 10",
    }
    model = exp.create_model("model",
                            path=test_path,
                            run_settings=run_settings_report_failure)

    ctrl.start(model, block=False)
    while not ctrl.finished(model):
        time.sleep(3)
    stat = ctrl.get_entity_status(model)
    assert(stat == constants.STATUS_FAILED)

def test_multiple_dpn():
    """Request and fail for a multiple dpn orchestrator running
       locally. We dont support this as we cannot launch a multi-prog
       job locally
    """
    exp_3 = Experiment("test", launcher="local")
    ctrl = exp_3._control
    O2 = exp_3.create_orchestrator(dpn=3)
    with pytest.raises(SSUnsupportedError):
        ctrl.start(O2)

