# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import pytest

from smartsim import Experiment
from smartsim.entity import Ensemble
from smartsim.entity.dbobject import FSModel
from smartsim.error.errors import SSUnsupportedError
from smartsim.log import get_logger
from smartsim.status import JobStatus

logger = get_logger(__name__)

should_run_tf = True
should_run_pt = True

# Check TensorFlow is available for tests
try:
    import tensorflow as tf
    from tensorflow import keras
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

    if pytest.test_device == "GPU":
        try:
            for device in tf.config.list_physical_devices("GPU"):
                tf.config.set_logical_device_configuration(
                    device, [tf.config.LogicalDeviceConfiguration(memory_limit=5_000)]
                )
        except:
            logger.warning("Could not set TF max memory limit for GPU")

should_run_tf &= (
    "tensorflow" in []
)  # todo: update test to replace installed_redisai_backends()

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


should_run_pt &= (
    "torch" in []
)  # todo: update test to replace installed_redisai_backends()


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
    n.eval()
    example_forward_input = torch.rand(1, 1, 28, 28)
    module = torch.jit.trace(n, example_forward_input)
    torch.jit.save(module, path + "/" + file_name)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_tf_fs_model(
    wlm_experiment, prepare_fs, single_fs, fileutils, test_dir, mlutils
):
    """Test TensorFlow FS Models on remote FS"""

    # Retrieve parameters from testing environment
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create RunSettings
    run_settings = wlm_experiment.create_run_settings(
        exe=sys.executable, exe_args=test_script
    )
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create Model
    smartsim_model = wlm_experiment.create_application("smartsim_model", run_settings)

    # Create feature store
    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)

    # Create and save ML model to filesystem
    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Add ML model to the SmartSim model
    smartsim_model.add_ml_model(
        "cnn",
        "TF",
        model=model,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
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
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
        tag="test",
    )

    logger.debug("The following ML models have been added:")
    for fs_model in smartsim_model._fs_models:
        logger.debug(fs_model)

    # Assert we have added both models
    assert len(smartsim_model._fs_models) == 2

    wlm_experiment.generate(smartsim_model)

    # Launch and check successful completion
    wlm_experiment.start(smartsim_model, block=True)
    statuses = wlm_experiment.get_status(smartsim_model)
    assert all(
        stat == JobStatus.COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.skipif(not should_run_pt, reason="Test needs PyTorch to run")
def test_pt_fs_model(
    wlm_experiment, prepare_fs, single_fs, fileutils, test_dir, mlutils
):
    """Test PyTorch FS Models on remote FS"""

    # Retrieve parameters from testing environment
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_pt_dbmodel_smartredis.py")

    # Create RunSettings
    run_settings = wlm_experiment.create_run_settings(
        exe=sys.executable, exe_args=test_script
    )
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create Model
    smartsim_model = wlm_experiment.create_applicationl("smartsim_model", run_settings)

    # Create feature store
    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)

    # Create and save ML model to filesystem
    save_torch_cnn(test_dir, "model1.pt")
    model_path = test_dir + "/model1.pt"

    # Add ML model to the SmartSim model
    smartsim_model.add_ml_model(
        "cnn",
        "TORCH",
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        tag="test",
    )

    logger.debug("The following ML models have been added:")
    for fs_model in smartsim_model._fs_models:
        logger.debug(fs_model)

    # Assert we have added both models
    assert len(smartsim_model._fs_models) == 1

    wlm_experiment.generate(smartsim_model)

    # Launch and check successful completion
    wlm_experiment.start(smartsim_model, block=True)
    statuses = wlm_experiment.get_status(smartsim_model)
    assert all(
        stat == JobStatus.COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_fs_model_ensemble(
    wlm_experiment, prepare_fs, single_fs, fileutils, test_dir, wlmutils, mlutils
):
    """Test FSModels on remote FS, with an ensemble"""

    # Retrieve parameters from testing environment
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create RunSettings
    run_settings = wlm_experiment.create_run_settings(
        exe=sys.executable, exe_args=test_script
    )
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create ensemble
    smartsim_ensemble = wlm_experiment.create_ensemble(
        "smartsim_model", run_settings=run_settings, replicas=2
    )

    # Create Model
    smartsim_model = wlm_experiment.create_application("smartsim_model", run_settings)

    # Create feature store
    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)

    # Create and save ML model to filesystem
    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Add the first ML model to all of the ensemble members
    smartsim_ensemble.add_ml_model(
        "cnn",
        "TF",
        model=model,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
    )

    # Add the second ML model individually to each SmartSim model
    for entity in smartsim_ensemble:
        entity.disable_key_prefixing()
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs2,
            outputs=outputs2,
        )

    # Add new ensemble member
    smartsim_ensemble.add_application(smartsim_model)

    # Add the second ML model to the newly added entity.  This is
    # because the test script runs both ML models for all entities.
    smartsim_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
    )

    # Assert we have added one model to the ensemble
    assert len(smartsim_ensemble._fs_models) == 1
    # Assert we have added two models to each entity
    assert all([len(entity._fs_models) == 2 for entity in smartsim_ensemble])

    wlm_experiment.generate(smartsim_ensemble)

    # Launch and check successful completion
    wlm_experiment.start(smartsim_ensemble, block=True)
    statuses = wlm_experiment.get_status(smartsim_ensemble)
    assert all(
        stat == JobStatus.COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_fs_model_tf(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Models on colocated fs (TensorFlow backend)"""

    # Set experiment name
    exp_name = "test-colocated-fs-model-tf"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create SmartSim Experience
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create colocated Model
    colo_model = exp.create_application("colocated_model", colo_settings)
    colo_model.colocate_fs_tcp(
        port=test_port, fs_cpus=1, debug=True, ifname=test_interface
    )

    # Create and save ML model to filesystem
    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Add ML models to the application
    colo_model.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
    )
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
    )

    # Assert we have added both models
    assert len(colo_model._fs_models) == 2

    exp.generate(colo_model)

    # Launch and check successful completion
    try:
        exp.start(colo_model, block=True)
        statuses = exp.get_status(colo_model)
        assert all(
            stat == JobStatus.COMPLETED for stat in statuses
        ), f"Statuses: {statuses}"
    finally:
        exp.stop(colo_model)


@pytest.mark.skipif(not should_run_pt, reason="Test needs PyTorch to run")
def test_colocated_fs_model_pytorch(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Models on colocated fs (PyTorch backend)"""

    # Set experiment name
    exp_name = "test-colocated-fs-model-pytorch"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_pt_dbmodel_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create colocated RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create colocated SmartSim Model
    colo_model = exp.create_application("colocated_model", colo_settings)
    colo_model.colocate_fs_tcp(
        port=test_port, fs_cpus=1, debug=True, ifname=test_interface
    )

    # Create and save ML model to filesystem
    save_torch_cnn(test_dir, "model1.pt")
    model_file = test_dir + "/model1.pt"

    # Add the ML model to the SmartSim Model
    colo_model.add_ml_model(
        "cnn",
        "TORCH",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added both models
    assert len(colo_model._fs_models) == 1

    exp.generate(colo_model)

    # Launch and check successful completion
    try:
        exp.start(colo_model, block=True)
        statuses = exp.get_status(colo_model)
        assert all(
            stat == JobStatus.COMPLETED for stat in statuses
        ), f"Statuses: {statuses}"
    finally:
        exp.stop(colo_model)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_fs_model_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """Test fsModel on colocated ensembles, first colocating fs,
    then adding fsModel.
    """

    # Set experiment name
    exp_name = "test-colocated-fs-model-ensemble"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings for colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create ensemble of two identical models
    colo_ensemble: Ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )

    # Create a third model with a colocated feature store
    colo_model = exp.create_application("colocated_model", colo_settings)
    colo_model.colocate_fs_tcp(
        port=test_port, fs_cpus=1, debug=True, ifname=test_interface
    )

    # Create and save the ML models to the filesystem
    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Colocate a feature store with the ensemble with two ensemble members
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i + 1, fs_cpus=1, debug=True, ifname=test_interface
        )
        # Add ML model to each ensemble member individual to test that they
        # do not conflict with models add to the Ensemble object
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs2,
            outputs=outputs2,
        )
        entity.disable_key_prefixing()

    # Test adding a model from Ensemble object
    colo_ensemble.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
        tag="test",
    )

    # Add a new model to the ensemble
    colo_ensemble.add_application(colo_model)

    # Add the ML model to SmartSim Model just added to the ensemble
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
    )

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all(
            stat == JobStatus.COMPLETED for stat in statuses
        ), f"Statuses: {statuses}"
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_fs_model_ensemble_reordered(fileutils, test_dir, wlmutils, mlutils):
    """Test fsModel on colocated ensembles, first adding the fsModel to the
    ensemble, then colocating fs.
    """

    # Set experiment name
    exp_name = "test-colocated-fs-model-ensemble-reordered"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create colocated RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create the ensemble of two identical SmartSim Model
    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )

    # Create colocated SmartSim Model
    colo_model = exp.create_application("colocated_model", colo_settings)

    # Create and save ML model to filesystem
    model_file, inputs, outputs = save_tf_cnn(test_dir, "model1.pb")
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
    )

    # Colocate a feature store with the first ensemble members
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i, fs_cpus=1, debug=True, ifname=test_interface
        )
        # Add ML models to each ensemble member to make sure they
        # do not conflict with other ML models
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_file2,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs2,
            outputs=outputs2,
        )
        entity.disable_key_prefixing()

    # Add another ensemble member
    colo_ensemble.add_application(colo_model)

    # Colocate a feature store with the new ensemble member
    colo_model.colocate_fs_tcp(
        port=test_port + len(colo_ensemble) - 1,
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )
    # Add a ML model to the new ensemble member
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
    )

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all(
            stat == JobStatus.COMPLETED for stat in statuses
        ), f"Statuses: {statuses}"
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_colocated_fs_model_errors(fileutils, test_dir, wlmutils, mlutils):
    """Test error when colocated fs model has no file."""

    # Set experiment name
    exp_name = "test-colocated-fs-model-error"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create colocated RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create colocated SmartSim Model
    colo_model = exp.create_application("colocated_model", colo_settings)
    colo_model.colocate_fs_tcp(
        port=test_port, fs_cpus=1, debug=True, ifname=test_interface
    )

    # Get and save TF model
    model, inputs, outputs = create_tf_cnn()

    # Check that an error is raised because in-memory models
    # are only supported for non-colocated deployments
    with pytest.raises(SSUnsupportedError):
        colo_model.add_ml_model(
            "cnn",
            "TF",
            model=model,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs,
            outputs=outputs,
        )

    # Create an ensemble with two identical replicas
    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )

    # Colocate a fs with each ensemble member
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i, fs_cpus=1, debug=True, ifname=test_interface
        )

    # Check that an error is raised because in-memory models
    # are only supported for non-colocated deployments
    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_ml_model(
            "cnn",
            "TF",
            model=model,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs,
            outputs=outputs,
        )

    # Check error is still thrown if an in-memory model is used
    # with a colocated deployment.  This test varies by adding
    # the SmartSIm model with a colocated feature store to the ensemble
    # after the ML model was been added to the ensemble.
    colo_settings2 = exp.create_run_settings(exe=sys.executable, exe_args=test_script)

    # Reverse order of fsModel and model
    colo_ensemble2 = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings2, replicas=2
    )
    colo_ensemble2.add_ml_model(
        "cnn",
        "TF",
        model=model,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
    )
    for i, entity in enumerate(colo_ensemble2):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_fs_tcp(
                port=test_port + i,
                fs_cpus=1,
                debug=True,
                ifname=test_interface,
            )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_application(colo_model)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TensorFlow to run")
def test_inconsistent_params_fs_model():
    """Test error when devices_per_node parameter>1 when devices is set to CPU in fsModel"""

    # Create and save ML model to filesystem
    model, inputs, outputs = create_tf_cnn()
    with pytest.raises(SSUnsupportedError) as ex:
        FSModel(
            "cnn",
            "TF",
            model=model,
            device="CPU",
            devices_per_node=2,
            first_device=0,
            tag="test",
            inputs=inputs,
            outputs=outputs,
        )
    assert (
        ex.value.args[0]
        == "Cannot set devices_per_node>1 if CPU is specified under devices"
    )


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
def test_fs_model_ensemble_duplicate(fileutils, test_dir, wlmutils, mlutils):
    """Test fsModels on remote fs, with an ensemble"""

    # Set experiment name
    exp_name = "test-fs-model-ensemble-duplicate"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1  # TF backend fails on multiple GPUs

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create ensemble
    smartsim_ensemble = exp.create_ensemble(
        "smartsim_ensemble", run_settings=run_settings, replicas=2
    )

    # Create Model
    smartsim_model = exp.create_application("smartsim_model", run_settings)

    # Create and save ML model to filesystem
    model, inputs, outputs = create_tf_cnn()
    model_file2, inputs2, outputs2 = save_tf_cnn(test_dir, "model2.pb")

    # Add the first ML model to all of the ensemble members
    smartsim_ensemble.add_ml_model(
        "cnn",
        "TF",
        model=model,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs,
        outputs=outputs,
    )

    # Attempt to add a duplicate ML model to Ensemble via Ensemble.add_ml_model()
    with pytest.raises(SSUnsupportedError) as ex:
        smartsim_ensemble.add_ml_model(
            "cnn",
            "TF",
            model=model,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs=inputs,
            outputs=outputs,
        )
    assert ex.value.args[0] == 'An ML Model with name "cnn" already exists'

    # Add same name ML model to a new SmartSim Model
    smartsim_model.add_ml_model(
        "cnn",
        "TF",
        model_path=model_file2,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=inputs2,
        outputs=outputs2,
    )

    # Attempt to add a duplicate ML model to Ensemble via Ensemble.add_application()
    with pytest.raises(SSUnsupportedError) as ex:
        smartsim_ensemble.add_application(smartsim_model)
    assert ex.value.args[0] == 'An ML Model with name "cnn" already exists'
