
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze_model(model, output_dir, file_name):

    if not file_name.endswith('.pb'):
        file_name = file_name + '.pb'

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
                    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    input_names = [x.name.split(':')[0] for x in frozen_func.inputs]
    output_names = [x.name.split(':')[0] for x in frozen_func.outputs]

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=output_dir,
                    name=file_name,
                    as_text=False)
    model_file_path = str(Path(output_dir, file_name).resolve())
    return model_file_path, input_names, output_names

