import pytest
import time

from glob import glob
from decorator import decorator
from shutil import rmtree, which, copyfile
from os import getcwd, listdir, path, environ, mkdir, remove

from smartsim import Experiment
from smartsim.utils import get_logger
from smartsim.control import Controller
from smartsim.tests.decorators import controller_test
from smartsim.error import SSUnsupportedError, SmartSimError

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")

# --- straightforward launching -------------------------------------

# experiment with non-clustered orchestrator
exp = Experiment("test")
ctrl = Controller()
ctrl.init_launcher("local")
run_settings = {
    "executable": "python",
    "exe_args": "sleep.py --time 2"
}
M1 = exp.create_model("m1", path=test_path, run_settings=run_settings)
M2 = exp.create_model("m2", path=test_path, run_settings=run_settings)
O1 = exp.create_orchestrator(path=test_path)

@controller_test
def test_ensemble():
    ctrl.start(ensembles=exp.ensembles)

# REQUIRES MANUAL SHUTDOWN OF LOCAL DATABASE
#@controller_test
#def test_ensemble_with_orc():
#    """Test launching a database locally with an ensemble"""
#    ctrl.start(ensembles=exp.ensembles, orchestrator=O1)

# --- Error handling ------------------------------------------------

def test_bad_num_orchestrators():
    """Test when the user uses create_orchestrator() and supplies
       a number of database nodes"""
    exp_2 = Experiment("test")
    O2 = exp_2.create_orchestrator(db_nodes=3)
    with pytest.raises(SmartSimError):
        ctrl.start(orchestrator=O2)

def test_multiple_dpn():
    """Request and fail for a multiple dpn orchestrator running
       locally. We dont support this as we cannot launch a multi-prog
       job locally
    """
    exp_2 = Experiment("test")
    O2 = exp_2.create_orchestrator(dpn=3)
    with pytest.raises(SSUnsupportedError):
        ctrl.start(orchestrator=O2)

# -- Unsupported commands by local launcher -------------------------

# in case the user tries commands that would with with Slurm, PBS or Capsules
# make sure that we throw the correct error.

# These represent commands we will hopefully leverage OS level mechanisms
# for to provide similar information to that of a WLM


def test_get_alloc():
    with pytest.raises(SSUnsupportedError):
        ctrl.get_allocation()

def test_accept_alloc():
    with pytest.raises(SSUnsupportedError):
        ctrl.add_allocation(111111)

def test_get_status():
    print(type(ctrl._jobs._launcher))
    with pytest.raises(SSUnsupportedError):
        ctrl.get_ensemble_status(exp.ensembles[0])

def test_stop():
    with pytest.raises(SSUnsupportedError):
        ctrl.stop(ensembles=exp.ensembles[0])

def test_is_finished():
    with pytest.raises(SSUnsupportedError):
        ctrl.poll(3, False, True)

def test_finished():
    with pytest.raises(SSUnsupportedError):
        ctrl.finished(exp.ensembles[0])

def test_release():
    with pytest.raises(SSUnsupportedError):
        ctrl.release(alloc_id=111111)
