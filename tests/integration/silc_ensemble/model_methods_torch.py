import os
import io

import numpy as np
import torch
import torch.nn as nn

import argparse

from silc import Client

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
    example_forward_input = torch.ones(1, 1, 3, 3)

    # Trace a module (implicitly traces `forward`) and construct a
    # `ScriptModule` with a single `forward` method
    module = torch.jit.trace(n, example_forward_input)

    # save model into an in-memory buffer then string
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    str_model = buffer.getvalue()
    return str_model


if __name__ == "__main__":
    # get model and set into database
    model = create_torch_cnn()
    c = Client(False)
    c.set_model("torch_cnn", model, "TORCH")

    keyout = os.getenv("SSKEYOUT")
    m = 1 if keyout == 'example_0' else -1

    keyin = os.getenv("SSKEYIN")

    # setup input tensor
    data = torch.ones(1, 1, 3, 3).numpy()*m
    c.put_tensor("torch_cnn_input", data)

    c.poll_key("torch_cnn_input", 100, 100)
    other_input = c.get_tensor("torch_cnn_input")

    assert np.sum(other_input+data)==0.0

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    c.poll_key("torch_cnn_output", 100, 100)
    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)
