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

def create_tf_cnn():
        """Create an in-memory Keras CNN for example purposes

        """
        from smartsim.ml.tf import serialize_model
        n = Net()
        input_shape = (3,3,1)
        inputs = Input(input_shape)
        outputs = n(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

        return serialize_model(model)

# Serialize and save TF model
model, inputs, outputs = create_tf_cnn()

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
ensemble_settings = exp.create_run_settings(exe="path/to/example_simulation_program")

# Initialize a Model object
ensemble_instance = exp.create_ensemble("ensemble_name", ensemble_settings)

# Attach the in-memory ML model to the SmartSim Ensemble
ensemble_instance.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)