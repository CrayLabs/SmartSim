import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_openmpi_lsf(wlmutils, fileutils):
    launcher = wlmutils.get_test_launcher()
    if launcher != "lsf":
        pytest.skip("Test only runs on systems with LSF as WLM")
    exp_name = "test-launch-openmpi-lsf"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", script, "mpirun")
    settings.set_cpus_per_task(1)
    settings.set_tasks(1)

    model = exp.create_model("ompi-model", path=test_dir, run_settings=settings)
    exp.start(model, block=True)
    statuses = exp.get_status(model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
