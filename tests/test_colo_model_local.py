import sys

from smartsim import Experiment, status



def test_launch_colocated_model(fileutils):
    """Test the launch of a model with a colocated database and local launcher"""

    exp_name = "test-launch-colocated-model-with-restart"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir(exp_name)
    sr_test_script = fileutils.get_test_conf_path("send_data_local_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=6780,
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo"
    )

    # assert model will launch with colocated db
    assert(colo_model.colocated)

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # test restarting the colocated model

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
