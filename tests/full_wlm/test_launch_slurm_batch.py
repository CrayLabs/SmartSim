import pytest

from smartsim import Experiment, constants
from smartsim.settings import SbatchSettings

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Test is only for Slurm WLM systems")


def test_batch_ensemble(fileutils, wlmutils):
    """Test the launch of a manually constructed batch ensemble"""

    exp_name = "test-slurm-batch-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    batch = SbatchSettings(nodes=2, time="00:01:00")
    ensemble = exp.create_ensemble("batch-ens", batch_settings=batch)
    ensemble.add_model(M1)
    ensemble.add_model(M2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])


def test_batch_ensemble_replicas(fileutils, wlmutils):
    exp_name = "test-slurm-batch-ensemble-replicas"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")

    batch = SbatchSettings(nodes=2, time="00:01:00")
    ensemble = exp.create_ensemble(
        "batch-ens-replicas", batch_settings=batch, run_settings=settings, replicas=2
    )
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == constants.STATUS_COMPLETED for stat in statuses])
