from smartsim import Experiment
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input

class Net(keras.Model):
        def __init__(self):
            super(Net, self).__init__(name="cnn")
            self.conv = Conv2D(1, 3, 1)

        def call(self, x):
            y = self.conv(x)
            return y

def save_tf_cnn(path, file_name):
    """Create a Keras CNN and save to file for example purposes"""
    from smartsim.ml.tf import freeze_model

    n = Net()
    input_shape = (3, 3, 1)
    n.build(input_shape=(None, *input_shape))
    inputs = Input(input_shape)
    outputs = n(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

    return freeze_model(model, path, file_name)

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="path/to/example_simulation_program")

# Initialize a Model object
model_instance = exp.create_model("model_name", model_settings)

# Get and save TF model
model_file, inputs, outputs = save_tf_cnn(model_instance.path, "model.pb")

# Attach the from file ML model to the SmartSim Model
model_instance.add_ml_model(name="cnn", backend="TF", model_path=model_file, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)