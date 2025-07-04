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

import typing as t
from pathlib import Path

import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (  # type: ignore[import-not-found,unused-ignore]
    convert_variables_to_constants_v2,
)


def freeze_model(
    model: keras.Model, output_dir: str, file_name: str
) -> t.Tuple[str, t.List[str], t.List[str]]:
    """Freeze a Keras or TensorFlow Graph

    to use a Keras or TensorFlow model in SmartSim, the model
    must be frozen and the inputs and outputs provided to the
    smartredis.client.set_model_from_file() method.

    This utiliy function provides everything users need to take
    a trained model and put it inside an ``orchestrator`` instance

    :param model: TensorFlow or Keras model
    :param output_dir: output dir to save model file to
    :param file_name: name of model file to create
    :return: path to model file, model input layer names, model output layer names
    """
    # TODO figure out why layer names don't match up to
    # specified name in Model init.

    if not file_name.endswith(".pb"):
        file_name = file_name + ".pb"

    full_model = tf.function(model)
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)  # type: ignore[no-untyped-call,unused-ignore]
    frozen_func.graph.as_graph_def()

    input_names = [x.name.split(":")[0] for x in frozen_func.inputs]
    output_names = [x.name.split(":")[0] for x in frozen_func.outputs]

    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=output_dir,
        name=file_name,
        as_text=False,
    )
    model_file_path = str(Path(output_dir, file_name).resolve())
    return model_file_path, input_names, output_names


def serialize_model(model: keras.Model) -> t.Tuple[str, t.List[str], t.List[str]]:
    """Serialize a Keras or TensorFlow Graph

    to use a Keras or TensorFlow model in SmartSim, the model
    must be frozen and the inputs and outputs provided to the
    smartredis.client.set_model() method.

    This utiliy function provides everything users need to take
    a trained model and put it inside an ``orchestrator`` instance.

    :param model: TensorFlow or Keras model
    :return: serialized model, model input layer names, model output layer names
    """

    full_model = tf.function(model)
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)  # type: ignore[no-untyped-call,unused-ignore]
    frozen_func.graph.as_graph_def()

    input_names = [x.name.split(":")[0] for x in frozen_func.inputs]
    output_names = [x.name.split(":")[0] for x in frozen_func.outputs]

    model_serialized = frozen_func.graph.as_graph_def().SerializeToString(
        deterministic=True
    )

    return model_serialized, input_names, output_names
