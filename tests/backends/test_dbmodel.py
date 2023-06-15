# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import time

import pytest

import smartsim
from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.error.errors import SSUnsupportedError

should_run_tf = True
should_run_pt = True

# Check TensorFlow is available for tests
try:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Conv2D, Input
except ImportError:
    should_run_tf = False
else:

    class Net(keras.Model):
        def __init__(self):
            super(Net, self).__init__(name="cnn")
            self.conv = Conv2D(1, 3, 1)

        def call(self, x):
            y = self.conv(x)
            return y


should_run_tf &= "tensorflow" in installed_redisai_backends()

# Check if PyTorch is available for tests
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    should_run_pt = False
else:
    # Simple MNIST in PyTorch
    class PyTorchNet(nn.Module):
        def __init__(self):
            super(PyTorchNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


should_run_pt &= "torch" in installed_redisai_backends()


def save_tf_cnn(path, file_name):
    """Create a Keras CNN for testing purposes"""
    from smartsim.ml.tf import freeze_model

    n = Net()
    input_shape = (3, 3, 1)
    n.build(input_shape=(None, *input_shape))
    inputs = Input(input_shape)
    outputs = n(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

    return freeze_model(model, path, file_name)


def create_tf_cnn():
    """Create a Keras CNN for testing purposes"""
    from smartsim.ml.tf import serialize_model

    n = Net()
    input_shape = (3, 3, 1)
    inputs = Input(input_shape)
    outputs = n(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

    return serialize_model(model)


def save_torch_cnn(path, file_name):
    n = PyTorchNet()
    example_forward_input = torch.rand(1, 1, 28, 28)
    module = torch.jit.trace(n, example_forward_input)
    torch.jit.save(module, path + "/" + file_name)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_tf_db_model(fileutils, wlmutils, mlutils):
    """Test TensorFlow DB Models on remote DB"""

    exp_name = "test-tf-db-model"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=wlmutils.get_test_port(), interface="lo")
    exp.generate(db)

    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    smartsim_model.add_ml_model(
        "cnn",
        "TF",
        model=model,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs,
        outputs=outputs,
        tag="test",
    )
    smartsim_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs2,
        outputs=outputs2,
        tag="test",
    )

    for db_model in smartsim_model._db_models:
        print(db_model)

    # Assert we have added both models
    assert len(smartsim_model._db_models) == 2

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_pt, reason="Test needs PyTorch to run")
def test_pt_db_model(fileutils, wlmutils, mlutils):
    """Test PyTorch DB Models on remote DB"""

    exp_name = "test-pt-db-model"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_pt_dbmodel_smartredis.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=wlmutils.get_test_port(), interface="lo")
    exp.generate(db)

    save_torch_cnn(test_dir, "model1.pt")
    model_path = test_dir + "/model1.pt"

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    smartsim_model.add_ml_model(
        "cnn",
        "TORCH",
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        tag="test",
    )

    for db_model in smartsim_model._db_models:
        print(db_model)

    # Assert we have added both models
    assert len(smartsim_model._db_models) == 1

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_db_model_ensemble(fileutils, wlmutils, mlutils):
    """Test DBModels on remote DB, with an ensemble"""

    exp_name = "test-db-model-ensemble"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    smartsim_ensemble = exp.create_ensemble(
        "smartsim_model", run_settings=run_settings, replicas=2
    )
    smartsim_ensemble.set_path(test_dir)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=wlmutils.get_test_port(), interface="lo")
    exp.generate(db)

    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    smartsim_ensemble.add_ml_model(
        "cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs
    )

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    for entity in smartsim_ensemble:
        entity.disable_key_prefixing()
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            inputs=inputs2,
            outputs=outputs2,
        )

    # Ensemble must add all available DBModels to new entity
    smartsim_ensemble.add_model(smartsim_model)
    smartsim_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs2,
        outputs=outputs2,
    )

    # Assert we have added one model to the ensemble
    assert len(smartsim_ensemble._db_models) == 1
    # Assert we have added two models to each entity
    assert all([len(entity._db_models) == 2 for entity in smartsim_ensemble])

    exp.start(db, smartsim_ensemble, block=True)
    statuses = exp.get_status(smartsim_ensemble)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_db_model_tf(fileutils, wlmutils, mlutils):
    """Test DB Models on colocated DB (TensorFlow backend)"""

    exp_name = "test-colocated-db-model-tf"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    colo_model.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs,
        outputs=outputs
    )
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs2,
        outputs=outputs2,
    )

    # Assert we have added both models
    assert len(colo_model._db_models) == 2

    exp.start(colo_model, block=False)

    completed = False
    timeout = 90
    check_interval = 5
    while timeout and not completed:
        timeout -= check_interval
        time.sleep(check_interval)
        statuses = exp.get_status(colo_model)
        if all([stat == status.STATUS_COMPLETED for stat in statuses]):
            completed = True

    if not completed:
        exp.stop(colo_model)
        assert False


@pytest.mark.skipif(not should_run_pt, reason="Test needs PyTorch to run")
def test_colocated_db_model_pytorch(fileutils, wlmutils, mlutils):
    """Test DB Models on colocated DB (PyTorch backend)"""

    exp_name = "test-colocated-db-model-pytorch"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_pt_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    save_torch_cnn(test_dir, "model1.pt")
    model_file = test_dir + "/model1.pt"
    colo_model.add_ml_model("cnn",
                            "TORCH",
                            model_path=model_file,
                            device=test_device,
                            devices_per_node=test_num_gpus)

    # Assert we have added both models
    assert len(colo_model._db_models) == 1

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_db_model_ensemble(fileutils, wlmutils, mlutils):
    """Test DBModel on colocated ensembles, first colocating DB,
    then adding DBModel.
    """

    exp_name = "test-colocated-db-model-ensemble"

    # get test setup
    test_dir = fileutils.make_test_dir()
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )
        # Test that models added individually do not conflict with enemble ones
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            inputs=inputs2,
            outputs=outputs2,
        )

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs,
        outputs=outputs,
        tag="test",
    )

    # Ensemble should add all available DBModels to new model
    colo_ensemble.add_model(colo_model)
    colo_model.colocate_db(
        port=wlmutils.get_test_port() + len(colo_ensemble),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs2,
        outputs=outputs2,
    )

    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_db_model_ensemble_reordered(fileutils, wlmutils, mlutils):
    """Test DBModel on colocated ensembles, first adding the DBModel to the
    ensemble, then colocating DB.
    """

    exp_name = "test-colocated-db-model-ensemble-reordered"

    # get test setup
    test_dir = fileutils.make_test_dir()
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model(
        "cnn", "TF", model_path=model_file, device="CPU", inputs=inputs, outputs=outputs
    )

    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )
        # Test that models added individually do not conflict with enemble ones
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            inputs=inputs2,
            outputs=outputs2,
        )

    # Ensemble should add all available DBModels to new model
    colo_ensemble.add_model(colo_model)
    colo_model.colocate_db(
        port=wlmutils.get_test_port() + len(colo_ensemble),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        inputs=inputs2,
        outputs=outputs2,
    )

    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_db_model_errors(fileutils, wlmutils, mlutils):
    """Test error when colocated db model has no file."""

    exp_name = "test-colocated-db-model-error"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    model, inputs, outputs = create_tf_cnn()

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()

    with pytest.raises(SSUnsupportedError):
        colo_model.add_ml_model(
            "cnn", "TF", model=model, device=test_device,
            devices_per_node=test_num_gpus, inputs=inputs, outputs=outputs
        )

    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_ml_model(
            "cnn", "TF", model=model, device=test_device,
            devices_per_node=test_num_gpus, inputs=inputs, outputs=outputs
        )

    # Check errors for reverse order of DBModel addition and DB colocation
    # create colocated model
    colo_settings2 = exp.create_run_settings(
        exe=sys.executable, exe_args=sr_test_script
    )

    # Reverse order of DBModel and model
    colo_ensemble2 = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings2, replicas=2
    )
    colo_ensemble2.set_path(test_dir)
    colo_ensemble2.add_ml_model(
        "cnn", "TF", model=model, device=test_device,
            devices_per_node=test_num_gpus, inputs=inputs, outputs=outputs
    )
    for i, entity in enumerate(colo_ensemble2):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_db(
                port=wlmutils.get_test_port() + i,
                db_cpus=1,
                limit_app_cpus=False,
                debug=True,
                ifname="lo",
            )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_model(colo_model)
