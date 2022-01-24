import time

import pytest

from smartsim import Experiment, status

"""
Test Stopping launched entities.

These tests will have their run settings automatically created
by the experiment which will choose the run_command so runtime may vary.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_stop_entity(fileutils, wlmutils):
    exp_name = "test-launch-stop-model"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=10")
    settings.set_tasks(1)
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=False)
    time.sleep(5)
    exp.stop(M1)
    assert M1.name in exp._control._jobs.completed
    assert exp.get_status(M1)[0] == status.STATUS_CANCELLED


def test_stop_entity_list(fileutils, wlmutils):

    exp_name = "test-launch-stop-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=10")
    settings.set_tasks(1)

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=False)
    time.sleep(5)
    exp.stop(ensemble)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
    assert all([m.name in exp._control._jobs.completed for m in ensemble])
