
import sys
import pytest
import numpy as np
from smartredis import Client
from smartredis.error import RedisReplyError

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.database import Orchestrator

try:
    from smartsim.tf import freeze_model
    from tensorflow import keras
except (ImportError, SmartSimError):
    pass

pytestmark = pytest.mark.skipif(
    ("tensorflow" not in sys.modules),
    reason="requires TensorFlow",
)
def create_tf_mnist_model():

    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(28, 28), name="input"),
        keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
        keras.layers.Dense(128, activation="relu", name="dense"),
        keras.layers.Dense(10, activation="softmax", name="output")],
                             name="FCN")

    # Compile model with optimizer
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def test_keras_model(fileutils):

    exp_name = "test_keras_model"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)


    model = create_tf_mnist_model()
    model_path, inputs, outputs = freeze_model(model, test_dir, "mnist.pb")

    db = Orchestrator(port=6780)
    db.set_path(test_dir)
    exp.start(db)

    test_status = True
    try:
        # connect a client to the database
        client = Client(address="127.0.0.1:6780",
                        cluster=False)

        client.set_model_from_file("tf_mnist", model_path, "TF",
                                   device="CPU", inputs=inputs, outputs=outputs)

        mnist_image = np.random.rand(1, 28, 28).astype(np.float32)
        client.put_tensor("mnist_input", mnist_image)
        client.run_model("tf_mnist", "mnist_input", "mnist_output")

        pred = client.get_tensor("mnist_output")
        print(pred)
        assert(len(pred[0]) == 10)

    except RedisReplyError as e:
        print("Caught a database error")
        print(e)
        test_status = False

    finally:
        exp.stop(db)
        assert(test_status)