import pytest

from smartsim import Experiment, constants
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError
from smartsim.settings import RunSettings

"""
Test the launch errors with local launcher
"""


def test_model_failure(fileutils):
    exp_name = "test-model-failure"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("bad.py")
    settings = RunSettings("python", f"{script} --time=3")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=True)
    statuses = exp.get_status(M1)
    assert all([stat == constants.STATUS_FAILED for stat in statuses])


def test_orchestrator_relaunch(fileutils):
    """Test error when users try to launch second orchestrator"""
    exp_name = "test-orc-error-on-relaunch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    orc = Orchestrator(port=6780)
    orc.set_path(test_dir)
    orc_1 = Orchestrator(port=6790)
    orc_1.set_path(test_dir)

    exp.start(orc)
    with pytest.raises(SmartSimError):
        exp.start(orc_1)

    exp.stop(orc)
