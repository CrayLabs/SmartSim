from copy import deepcopy

import pytest

from smartsim import Experiment, status

"""
Test the launch of simple entity types on pre-existing allocations.

All entities will obtain the allocation from the environment of the
user

Each of the tests below will have their RunSettings automatically
created which means that these tests may vary in run command that
is used.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_models(fileutils, wlmutils):
    exp_name = "test-models-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=5")
    settings.set_tasks(1)

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=deepcopy(settings))

    exp.start(M1, M2, block=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_ensemble(fileutils, wlmutils):
    exp_name = "test-ensemble-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=5")
    settings.set_tasks(1)

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_summary(fileutils, wlmutils):
    """Fairly rudimentary test of the summary dataframe"""

    exp_name = "test-launch-summary"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    sleep = fileutils.get_test_conf_path("sleep.py")
    bad = fileutils.get_test_conf_path("bad.py")

    sleep_settings = exp.create_run_settings("python", f"{sleep} --time=3")
    sleep_settings.set_tasks(1)
    bad_settings = exp.create_run_settings("python", f"{bad} --time=6")
    bad_settings.set_tasks(1)

    sleep = exp.create_model("sleep", path=test_dir, run_settings=sleep_settings)
    bad = exp.create_model("bad", path=test_dir, run_settings=bad_settings)

    # start and poll
    exp.start(sleep, bad)
    assert exp.get_status(bad)[0] == status.STATUS_FAILED
    assert exp.get_status(sleep)[0] == status.STATUS_COMPLETED

    summary_str = exp.summary(format="plain")
    print(summary_str)

    rows = [s.split() for s in summary_str.split("\n")]
    headers = ["Index"] + rows.pop(0)

    row = dict(zip(headers, rows[0]))
    assert sleep.name == row["Name"]
    assert sleep.type == row["Entity-Type"]
    assert 0 == int(row["RunID"])
    assert 0 == int(row["Returncode"])

    row_1 = dict(zip(headers, rows[1]))
    assert bad.name == row_1["Name"]
    assert bad.type == row_1["Entity-Type"]
    assert 0 == int(row_1["RunID"])
    assert 0 != int(row_1["Returncode"])
