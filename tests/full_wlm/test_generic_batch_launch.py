from time import sleep

import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_batch_ensemble(fileutils, wlmutils):
    """Test the launch of a manually constructed batch ensemble"""

    exp_name = "test-batch-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")
    if wlmutils.get_test_launcher() == "lsf":
        batch.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        batch.set_account(wlmutils.get_test_account())
        batch.set_queue("debug-flat-quad")
    ensemble = exp.create_ensemble("batch-ens", batch_settings=batch)
    ensemble.add_model(M1)
    ensemble.add_model(M2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_batch_ensemble_replicas(fileutils, wlmutils):
    exp_name = "test-batch-ensemble-replicas"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")
    if wlmutils.get_test_launcher() == "lsf":
        batch.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        # As Cobalt won't allow us to run two
        # jobs in the same debug queue, we need
        # to make sure the previous test's one is over
        sleep(30)
        batch.set_account(wlmutils.get_test_account())
        batch.set_queue("debug-flat-quad")
    ensemble = exp.create_ensemble(
        "batch-ens-replicas", batch_settings=batch, run_settings=settings, replicas=2
    )
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
