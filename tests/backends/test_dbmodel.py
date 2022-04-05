import sys
import pytest

from smartsim import Experiment, status
import smartsim
from smartsim._core.utils import installed_redisai_backends
from smartsim.error.errors import SSUnsupportedError

should_run = True

try:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Conv2D, Input
except ImportError:
    should_run = False

should_run &= "tensorflow" in installed_redisai_backends()

class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__(name="cnn")
        self.conv = Conv2D(1, 3, 1)

    def call(self, x):
        y = self.conv(x)
        return y


def save_tf_cnn(path, file_name):
    """Create a Keras CNN for testing purposes

    """
    from smartsim.ml.tf import freeze_model
    n = Net()
    input_shape = (3,3,1)
    n.build(input_shape=(None,*input_shape))
    inputs = Input(input_shape)
    outputs = n(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

    return freeze_model(model, path, file_name)


def create_tf_cnn():
    """Create a Keras CNN for testing purposes

    """
    from smartsim.ml.tf import serialize_model
    n = Net()
    input_shape = (3,3,1)
    inputs = Input(input_shape)
    outputs = n(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

    return serialize_model(model)


@pytest.mark.skipif(not should_run, reason="Test needs TF to run")
def test_colocated_db_model(fileutils):
    """Test DB Models on colocated DB"""

    exp_name = "test-colocated-db-model"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

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

    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    colo_model.add_ml_model("cnn", "TF", model_path=model_file, device="CPU", inputs=inputs, outputs=outputs)
    colo_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    # Assert we have added both models
    assert(len(colo_model._db_models) == 2)

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

@pytest.mark.skipif(not should_run, reason="Test needs TF to run")
def test_db_model(fileutils):
    """Test DB Models on remote DB"""

    exp_name = "test-db-model"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=6780, interface="lo")

    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    smartsim_model.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)
    smartsim_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    for db_model in smartsim_model._db_models:
        print(db_model)

    # Assert we have added both models
    assert(len(smartsim_model._db_models) == 2)

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run or not "tensorflow" in installed_redisai_backends(), reason="Test needs TF to run")
def test_colocated_db_model_error(fileutils):
    """Test error when colocated db model has no file."""

    exp_name = "test-colocated-db-model-error"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

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

    model, inputs, outputs = create_tf_cnn()

    colo_model.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)

    with pytest.raises(SSUnsupportedError):
        exp.start(colo_model, block=True)


