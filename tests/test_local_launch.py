from smartsim import Experiment, status

"""
Test the launch of simple entity types with local launcher
"""


def test_models(fileutils):
    exp_name = "test-models-local-launch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=3")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    exp.start(M1, M2, block=True, summary=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_ensemble(fileutils):
    exp_name = "test-ensemble-launch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=3")

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True, summary=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
