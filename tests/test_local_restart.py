from smartsim import Experiment, status

"""
Test restarting ensembles and models.
"""


def test_restart(fileutils):

    exp_name = "test-models-local-restart"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=3")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=True)
    statuses = exp.get_status(M1)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # restart the model
    exp.start(M1, block=True)
    statuses = exp.get_status(M1)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_ensemble(fileutils):
    exp_name = "test-ensemble-restart"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=3")

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # restart the ensemble
    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
