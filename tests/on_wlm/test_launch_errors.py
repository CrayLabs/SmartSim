import time

import pytest

from smartsim import Experiment, constants
from smartsim.error import LauncherError, SmartSimError

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_failed_status(fileutils, wlmutils):
    """Test when a failure occurs deep into model execution"""

    exp_name = "test-report-failure"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("bad.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=7")

    model = exp.create_model("bad-model", path=test_dir, run_settings=settings)

    exp.start(model, block=False)
    while not exp.finished(model):
        time.sleep(2)
    status = exp.get_status(model)
    assert status[0] == constants.STATUS_FAILED


def test_bad_run_command_args(fileutils, wlmutils):
    """Should fail because of incorrect arguments given to the
    run command

    This test ensures that we catch immediate failures
    """
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip(f"Only fails with slurm. Launcher is {launcher}")

    exp_name = "test-bad-run-command-args"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("bad.py")

    # this argument will get turned into an argument for the run command
    # of the specific WLM of the system.
    settings = wlmutils.get_run_settings(
        "python", f"{script} --time=5", badarg="bad-arg"
    )

    model = exp.create_model("bad-model", path=test_dir, run_settings=settings)

    with pytest.raises(SmartSimError):
        exp.start(model)
