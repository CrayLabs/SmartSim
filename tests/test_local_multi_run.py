from smartsim import Experiment, status

"""
Test the launch of simple entity types with local launcher
"""


def test_models(fileutils):
    exp_name = "test-models-local-launch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=5")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    exp.start(M1, block=False)
    statuses = exp.get_status(M1)
    assert all([stat != status.STATUS_FAILED for stat in statuses])

    # start another while first model is running
    exp.start(M2, block=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
