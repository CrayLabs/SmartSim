import os

import numpy as np
from smartredis import Client
from tensorflow import keras

from smartsim.ml.tf import freeze_model


def create_tf_mnist_model():

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


def run(device):

    model = create_tf_mnist_model()
    model_path, inputs, outputs = freeze_model(model, os.getcwd(), "mnist.pb")

    client = Client(cluster=False)
    client.set_model_from_file(
        "tf_mnist", model_path, "TF", device=device, inputs=inputs, outputs=outputs
    )

    mnist_image = np.random.rand(1, 28, 28).astype(np.float32)
    client.put_tensor("mnist_input", mnist_image)
    client.run_model("tf_mnist", "mnist_input", "mnist_output")

    pred = client.get_tensor("mnist_output")
    print(pred)
    assert len(pred[0]) == 10


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keras test Script")
    parser.add_argument(
        "--device", type=str, default="CPU", help="device type for model execution"
    )
    args = parser.parse_args()
    run(args.device)
