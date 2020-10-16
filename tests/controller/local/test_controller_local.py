import pytest
import time

from glob import glob
from decorator import decorator
from shutil import rmtree, which, copyfile
from os import getcwd, listdir, path, environ, mkdir, remove

from smartsim import Experiment
from smartsim.utils import get_logger
from smartsim.control import Controller
from smartsim.utils.test.decorators import controller_test
from smartsim.error import SSUnsupportedError, SmartSimError

# create some entities for testing
test_path = path.join(getcwd(),  "./controller_test/")

# --- straightforward launching -------------------------------------

exp = Experiment("test")
ctrl = Controller()
ctrl.init_launcher("local")
local_run_settings = {
    "executable": "python",
    "exe_args": "sleep.py --time 2"
}
M1 = exp.create_model("m1", path=test_path, run_settings=local_run_settings)
M2 = exp.create_model("m2", path=test_path, run_settings=local_run_settings)

@controller_test
def test_ensemble():
    ctrl.start(M1, M2)

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
        ctrl.start(O2)

def test_multiple_dpn():
    """Request and fail for a multiple dpn orchestrator running
       locally. We dont support this as we cannot launch a multi-prog
       job locally
    """
    exp_2 = Experiment("test")
    O2 = exp_2.create_orchestrator(dpn=3)
    with pytest.raises(SSUnsupportedError):
        ctrl.start(O2)

