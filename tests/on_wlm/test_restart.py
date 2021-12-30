from copy import deepcopy

import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_restart(fileutils, wlmutils):

    exp_name = "test-restart"
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

    exp.start(M1, M2, block=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # TODO add job history check here.
