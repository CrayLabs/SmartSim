import time

import pytest

from smartsim import Experiment, constants
from smartsim.settings.settings import RunSettings

"""
Test the launch and stop of models and ensembles using base
RunSettings while on WLM.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_model_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in  ["pbs", "slurm", "cobalt"]:
        pytest.skip("Test only runs on systems with PBSPro, Slurm, or Cobalt as WLM")

    exp_name = "test-base-settings-model-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings1 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    settings2 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)

    # launch models twice to show that they can also be restarted
    for _ in range(2):
        exp.start(M1, M2, block=True)
        statuses = exp.get_status(M1, M2)
        assert all([stat == constants.STATUS_COMPLETED for stat in statuses])


def test_model_stop_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in  ["pbs", "slurm", "cobalt"]:
        pytest.skip("Test only runs on systems with PBSPro, Slurm, or Cobalt as WLM")

    exp_name = "test-base-settings-model-stop"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings1 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    settings2 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)

    # stop launched models
    exp.start(M1, M2, block=False)
    time.sleep(2)
    exp.stop(M1, M2)
    assert M1.name in exp._control._jobs.completed
    assert M2.name in exp._control._jobs.completed
    statuses = exp.get_status(M1, M2)
    assert all([stat == constants.STATUS_CANCELLED for stat in statuses])


def test_ensemble_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in  ["pbs", "slurm", "cobalt"]:
        pytest.skip("Test only runs on systems with PBSPro, Slurm, or Cobalt as WLM")

    exp_name = "test-base-settings-ensemble-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    ensemble = exp.create_ensemble("ensemble", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    # launch ensemble twice to show that it can also be restarted
    for _ in range(2):
        exp.start(ensemble, block=True)
        statuses = exp.get_status(ensemble)
        assert all([stat == constants.STATUS_COMPLETED for stat in statuses])


def test_ensemble_stop_on_wlm(fileutils, wlmutils):
    launcher = wlmutils.get_test_launcher()
    if launcher not in  ["pbs", "slurm", "cobalt"]:
        pytest.skip("Test only runs on systems with PBSPro, Slurm, or Cobalt as WLM")

    exp_name = "test-base-settings-ensemble-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    ensemble = exp.create_ensemble("ensemble", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    # stop launched ensemble
    exp.start(ensemble, block=False)
    time.sleep(2)
    exp.stop(ensemble)
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_CANCELLED for stat in statuses])
    assert all([m.name in exp._control._jobs.completed for m in ensemble])
