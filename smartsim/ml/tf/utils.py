from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def freeze_model(model, output_dir, file_name):
    """Freeze a Keras or TensorFlow Graph

    to use a Keras or TensorFlow model in SmartSim, the model
    must be frozen and the inputs and outputs provided to the
    smartredis.client.set_model_from_file() method.

    This utiliy function provides everything users need to take
    a trained model and put it inside an ``orchestrator`` instance

    :param model: TensorFlow or Keras model
    :type model: tf.Module
    :param output_dir: output dir to save model file to
    :type output_dir: str
    :param file_name: name of model file to create
    :type file_name: str
    :return: path to model file, model input layer names, model output layer names
    :rtype: str, list[str], list[str]
    """
    # TODO figure out why layer names don't match up to
    # specified name in Model init.

    if not file_name.endswith(".pb"):
        file_name = file_name + ".pb"

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)
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
