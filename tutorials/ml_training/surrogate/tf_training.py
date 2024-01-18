from smartsim.ml.tf import DynamicDataGenerator, serialize_model
from smartredis import Client, Dataset

from tensorflow import keras
from tensorflow.keras.layers import Input
from tf_model import DiffusionResNet

import time
import numpy as np


def create_dataset(idx, F):
    """Create SmartRedis Dataset containing multiple NumPy arrays
    to be stored at a single key within the database"""

    dataset = Dataset(f"ml_data_{idx}")
    dataset.add_tensor("steady", F)
    return dataset


def store_model(model, idx):
    serialized_model, inputs, outputs = serialize_model(model)
    client = Client(None, False)
    client.set_model(f"{model.name}_{idx}", serialized_model, "TF", "CPU", inputs=inputs, outputs=outputs)

def train_model(model, epochs):
    training_generator = DynamicDataGenerator(cluster=False, batch_size=50, shuffle=True, data_info_or_list_name="training_data")
    print("Compiling NN")

    initial_learning_rate = 0.01
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=80,
        decay_rate=0.9,
        staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-07)

    model.compile(optimizer=optimizer, loss="mean_absolute_error")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model.fit(training_generator, steps_per_epoch=None, 
                  epochs=epoch+1, initial_epoch=epoch, batch_size=training_generator.batch_size,
                  verbose=2)
        if (epoch+1)%10 == 0:
            store_model(model, epoch//10)

    print("Finished training", flush=True)


def upload_inference_examples(model, num_examples):
    client = Client(None, False)
    client.set_data_source("fd_simulation")

    for i in range(num_examples):
        ds = client.get_dataset(f"sim_data_{i}")
        u_init = np.expand_dims(ds.get_tensor("u_init"), [0, -1])
        frame = model(u_init, training=False).numpy().squeeze()
        ds = create_dataset(i, frame)
        client.put_dataset(ds)

    print("Finished upload")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finite Difference Simulation")
    parser.add_argument('--depth', type=int, default=4, 
                        help="Half depth of residual network")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of epochs to train the NN for")
    parser.add_argument('--delay', type=int, default=0, 
                        help="Seconds to wait before training")
    parser.add_argument('--size', type=int, default=100,
                        help='Size of sample side, each sample will be a (size, size, 1) image')

    args = parser.parse_args()
    input_shape = (args.size,args.size,1)
    diff_resnet = DiffusionResNet(input_shape, depth=args.depth)
    diff_resnet.build(input_shape=(None,*input_shape))
    diff_resnet.summary()
    inputs = Input(input_shape)
    outputs = diff_resnet(inputs)
    vae = keras.Model(inputs=inputs, outputs=outputs, name=diff_resnet.name)

    time.sleep(args.delay)
    train_model(vae, args.epochs)

    args = parser.parse_args()