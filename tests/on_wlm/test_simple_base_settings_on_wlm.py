import time

import pytest

from smartsim import Experiment, status
from smartsim.settings.settings import RunSettings

"""
Test the launch and stop of simple models and ensembles that use base
RunSettings while on WLM that do not include a run command

These tests will execute code (very light scripts) on the head node
as no run command will exec them out onto the compute nodes

the reason we test this is because we want to make sure that each
WLM launcher also supports the ability to run scripts on the head node
that also contain an invoking run command
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_simple_model_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in ["pbs", "slurm", "cobalt", "lsf"]:
        pytest.skip(
            "Test only runs on systems with LSF, PBSPro, Slurm, or Cobalt as WLM"
        )

    exp_name = "test-simplebase-settings-model-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = RunSettings("python", exe_args=f"{script} --time=5")
    M = exp.create_model("m", path=test_dir, run_settings=settings)

    # launch model twice to show that it can also be restarted
    for _ in range(2):
        exp.start(M, block=True)
        assert exp.get_status(M)[0] == status.STATUS_COMPLETED


def test_simple_model_stop_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in ["pbs", "slurm", "cobalt", "lsf"]:
        pytest.skip(
            "Test only runs on systems with LSF, PBSPro, Slurm, or Cobalt as WLM"
        )

    exp_name = "test-simplebase-settings-model-stop"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = RunSettings("python", exe_args=f"{script} --time=5")
    M = exp.create_model("m", path=test_dir, run_settings=settings)

    # stop launched model
    exp.start(M, block=False)
    time.sleep(2)
    exp.stop(M)
    assert M.name in exp._control._jobs.completed
    assert exp.get_status(M)[0] == status.STATUS_CANCELLED
