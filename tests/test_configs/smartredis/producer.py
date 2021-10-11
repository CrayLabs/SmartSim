import argparse
import io
import os

import numpy as np
import torch
import torch.nn as nn

from smartredis import Client


# taken from https://pytorch.org/docs/master/generated/torch.jit.trace.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)


def create_torch_cnn():
    """Create a torch CNN for testing purposes

    Jit traces the torch Module for storage in RedisAI
    """
    n = Net()
    producer_forward_input = torch.ones(1, 1, 3, 3)

    # Trace a module (implicitly traces `forward`) and construct a
    # `ScriptModule` with a single `forward` method
    module = torch.jit.trace(n, producer_forward_input)

    # save model into an in-memory buffer then string
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    str_model = buffer.getvalue()
    return str_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartRedis ensemble producer process."
    )
    parser.add_argument("--exchange", action="store_true")
    args = parser.parse_args()

    # get model and set into database
    model = create_torch_cnn()
    c = Client(False)
    c.set_model("torch_cnn", model, "TORCH")

    keyout = os.getenv("SSKEYOUT")
    keyin = os.getenv("SSKEYIN")

    assert keyout in ["producer_0", "producer_1"]

    if keyout == "producer_0":
        c.set_data_source("producer_1" if args.exchange else "producer_0")
        data = torch.ones(1, 1, 3, 3).numpy()
        data_other = -torch.ones(1, 1, 3, 3).numpy()
    elif keyout == "producer_1":
        c.set_data_source("producer_0" if args.exchange else "producer_1")
        data = -torch.ones(1, 1, 3, 3).numpy()
        data_other = torch.ones(1, 1, 3, 3).numpy()

    # setup input tensor
    c.put_tensor("torch_cnn_input", data)

    input_exists = c.poll_tensor("torch_cnn_input", 100, 100)
    assert input_exists

    other_input = c.get_tensor("torch_cnn_input")

    if args.exchange:
        assert np.all(other_input == data_other)
    else:
        assert np.all(other_input == data)

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    output_exists = c.poll_tensor("torch_cnn_output", 100, 100)
    assert output_exists

    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)
