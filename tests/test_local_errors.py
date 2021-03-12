from smartsim import Experiment, constants
from smartsim.settings import RunSettings

"""
Test the launch of simple entity types with local launcher
"""

def test_model_failure(fileutils):
    exp_name = "test-model-failure"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("bad.py")
    settings = RunSettings("python", f"{script} --time=3")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=True)
    statuses = exp.get_status(M1)
    assert all([stat == constants.STATUS_FAILED for stat in statuses])
