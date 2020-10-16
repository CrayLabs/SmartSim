
import pytest
import filecmp
from os import path, getcwd

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.utils.test.decorators import orchestrator_test_local


test_path = path.join(getcwd(),  "./orchestrator_test/")

# --- database reconnect ----------------------------------------------

@orchestrator_test_local
def test_db_reconnect_send_receive_local():

    exp_1_dir = "/".join((test_path,"exp_1"))
    exp_2_dir = "/".join((test_path,"exp_2"))

    sim_dict = {"executable": "python reconnect_sim.py" }
    exp_1 = Experiment("exp_1", launcher="local")
    O1 = exp_1.create_orchestrator(path=exp_1_dir)
    M1 = exp_1.create_model("M1", path=exp_1_dir, run_settings=sim_dict)
    exp_1.start()

    exp_2 = Experiment("exp_2", launcher="local")
    with pytest.raises(SmartSimError):
        O2 = exp_2.reconnect_orchestrator(exp_1_dir)

