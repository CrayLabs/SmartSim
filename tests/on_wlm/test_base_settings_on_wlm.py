import time

import pytest

from smartsim import Experiment, status

"""
Test the launch and stop of models and ensembles using base
RunSettings while on WLM.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_model_on_wlm(fileutils, wlmutils):
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
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_model_stop_on_wlm(fileutils, wlmutils):
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
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
