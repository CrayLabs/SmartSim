import sys
import pytest

from smartsim import Experiment, status
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
    exp.generate(db)

    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    smartsim_model.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs, tag="test")
    smartsim_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2, tag="test")

    for db_model in smartsim_model._db_models:
        print(db_model)

    # Assert we have added both models
    assert(len(smartsim_model._db_models) == 2)

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs TF to run")
def test_db_model_ensemble(fileutils):
    """Test DBModels on remote DB, with an ensemble"""

    exp_name = "test-db-model-ensemble"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    smartsim_ensemble = exp.create_ensemble("smartsim_model", run_settings=run_settings, replicas=2)
    smartsim_ensemble.set_path(test_dir)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=6780, interface="lo")
    exp.generate(db)

    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    smartsim_ensemble.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)
    
    for entity in smartsim_ensemble:
        entity.disable_key_prefixing()
        entity.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    # Ensemble must add all available DBModels to new entity
    smartsim_ensemble.add_model(smartsim_model)
    smartsim_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    # Assert we have added one model to the ensemble
    assert(len(smartsim_ensemble._db_models) == 1)
    # Assert we have added two models to each entity
    assert(all([len(entity._db_models)==2 for entity in smartsim_ensemble]))

    exp.start(db, smartsim_ensemble, block=True)
    statuses = exp.get_status(smartsim_ensemble)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


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
def test_colocated_db_model_ensemble(fileutils):
    """Test DBModel on colocated ensembles, first colocating DB,
    then adding DBModel.
    """

    exp_name = "test-colocated-db-model-ensemble"

    # get test setup
    test_dir = fileutils.make_test_dir()
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    colo_ensemble = exp.create_ensemble("colocated_ens", run_settings=colo_settings, replicas=2)
    colo_ensemble.set_path(test_dir)

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

    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=6780+i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo"
            )
        # Test that models added individually do not conflict with enemble ones
        entity.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model("cnn", "TF", model_path=model_file, device="CPU", inputs=inputs, outputs=outputs, tag="test")

    # Ensemble should add all available DBModels to new model
    colo_ensemble.add_model(colo_model)
    colo_model.colocate_db(
            port=6780+len(colo_ensemble),
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo"
            )
    colo_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)


    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs TF to run")
def test_colocated_db_model_ensemble_reordered(fileutils):
    """Test DBModel on colocated ensembles, first adding the DBModel to the
    ensemble, then colocating DB.
    """

    exp_name = "test-colocated-db-model-ensemble-reordered"

    # get test setup
    test_dir = fileutils.make_test_dir()
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    sr_test_script = fileutils.get_test_conf_path("run_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    colo_ensemble = exp.create_ensemble("colocated_ens", run_settings=colo_settings, replicas=2)
    colo_ensemble.set_path(test_dir)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model("cnn", "TF", model_path=model_file, device="CPU", inputs=inputs, outputs=outputs)

    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=6780+i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo"
            )
        # Test that models added individually do not conflict with enemble ones
        entity.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)


    # Ensemble should add all available DBModels to new model
    colo_ensemble.add_model(colo_model)
    colo_model.colocate_db(
            port=6780+len(colo_ensemble),
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo"
            )
    colo_model.add_ml_model("cnn2", "TF", model_path=model_file2, device="CPU", inputs=inputs2, outputs=outputs2)

    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs TF to run")
def test_colocated_db_model_errors(fileutils):
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

    with pytest.raises(SSUnsupportedError):
        colo_model.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)


    colo_ensemble = exp.create_ensemble("colocated_ens", run_settings=colo_settings, replicas=2)
    colo_ensemble.set_path(test_dir)
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=6780+i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo"
        )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)

    # Check errors for reverse order of DBModel addition and DB colocation
    # create colocated model
    colo_settings2 = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    # Reverse order of DBModel and model
    colo_ensemble2 = exp.create_ensemble("colocated_ens", run_settings=colo_settings2, replicas=2)
    colo_ensemble2.set_path(test_dir)
    colo_ensemble2.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)
    for i, entity in enumerate(colo_ensemble2):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_db(
                port=6780+i,
                db_cpus=1,
                limit_app_cpus=False,
                debug=True,
                ifname="lo"
                )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_model(colo_model)