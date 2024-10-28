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

import os
from pathlib import Path

import pytest

from smartsim.error import SmartSimError
from smartsim.status import JobStatus

tf_available = True
try:
    from tensorflow import keras

    from smartsim.ml.tf import freeze_model, serialize_model
except (ImportError, SmartSimError) as e:
    print(e)
    tf_available = False

tf_backend_available = (
    "tensorflow" in []
)  # todo: update test to replace installed_redisai_backends()


@pytest.mark.skipif(
    (not tf_backend_available) or (not tf_available),
    reason="Requires RedisAI TF backend",
)
def test_keras_model(wlm_experiment, prepare_fs, single_fs, mlutils, wlmutils):
    """This test needs two free nodes, 1 for the db and 1 for a keras model script

    this test can run on CPU/GPU by setting SMARTSIM_TEST_DEVICE=GPU
    Similarly, the test can execute on any launcher by setting SMARTSIM_TEST_LAUNCHER
    which is local by default.

    You may need to put CUDNN in your LD_LIBRARY_PATH if running on GPU
    """

    test_device = mlutils.get_test_device()
    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)

    run_settings = wlm_experiment.create_run_settings(
        "python", f"run_tf.py --device={test_device}"
    )

    if wlmutils.get_test_launcher() != "local":
        run_settings.set_tasks(1)
    model = wlm_experiment.create_application("tf_script", run_settings)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = Path(script_dir, "run_tf.py").resolve()
    model.attach_generator_files(to_copy=str(script_path))
    wlm_experiment.generate(model)

    wlm_experiment.start(model, block=True)

    # if model failed, test will fail
    model_status = wlm_experiment.get_status(model)[0]
    assert model_status != JobStatus.FAILED


def create_tf_model():
    model = keras.Sequential(
        layers=[
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="FCN",
    )

    # Compile model with optimizer
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


@pytest.mark.skipif(not tf_available, reason="Requires Tensorflow and Keras")
def test_freeze_model(test_dir):
    model = create_tf_model()
    model_path, inputs, outputs = freeze_model(model, test_dir, "mnist.pb")
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert Path(model_path).is_file()


@pytest.mark.skipif(not tf_available, reason="Requires Tensorflow and Keras")
def test_serialize_model():
    model = create_tf_model()
    model_serialized, inputs, outputs = serialize_model(model)
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(model_serialized) > 0
