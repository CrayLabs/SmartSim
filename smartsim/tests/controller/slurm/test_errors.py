import pytest
import time
from shutil import which
from os import getcwd, path, environ

from smartsim import Experiment
from smartsim.control import Controller
from smartsim.tests.decorators import controller_test
from smartsim.error import SmartSimError
# --- Setup ---------------------------------------------------

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")

ctrl = Controller()

def test_setup_alloc():
    """Not a test, just used to ensure that at test time, the
       allocation is added to the controller. This has to be a
       test because otherwise it will run on pytest startup.
    """
    if not which("srun"):
        pytest.skip()
    alloc_id = environ["TEST_ALLOCATION_ID"]
    ctrl.add_allocation(alloc_id)
    assert("TEST_ALLOCATION_ID" in environ)

# ----- Tests ---------------------------------------------------------

@controller_test
def test_failed_status():
    """Test when a failure occurs deep into model execution"""

    exp = Experiment("test_report_failure")
    alloc = get_alloc_id()

    run_settings_report_failure = {
        "ppn": 1,
        "nodes": 1,
        "executable": "python bad.py",
        "exe_args": "--time 10",
        "alloc": alloc
    }
    model = exp.create_model("model",
                            path=test_path,
                            run_settings=run_settings_report_failure)

    ctrl.start(exp.ensembles)
    while not ctrl.finished(model):
        time.sleep(3)
    status = ctrl.get_model_status(model)
    assert(status == "FAILED")

@controller_test
def test_start_no_allocs():
    """test when a user doesnt provide an allocation with entity run_settings"""

    exp = Experiment("test_no_alloc")

    run_settings_no_alloc = {
        "ppn": 1,
        "nodes": 1,
        "executable": "python bad.py",
        "exe_args": "--time 10"
    }
    model = exp.create_model("model_no_alloc",
                             path=test_path,
                             run_settings=run_settings_no_alloc)

    with pytest.raises(SmartSimError):
        ctrl.start(exp.ensembles)


@controller_test
def test_bad_release():
    """Test when a non-existant alloc_id is given to release"""
    with pytest.raises(SmartSimError):
        ctrl.release(alloc_id=111111)


# ------ Helper Functions ------------------------------------------

def get_alloc_id():
    alloc_id = environ["TEST_ALLOCATION_ID"]
    return alloc_id

