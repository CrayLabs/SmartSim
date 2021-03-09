import os
import io

import numpy as np
import torch
import torch.nn as nn

import argparse

from silc import Client, EntityType

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
    parser = argparse.ArgumentParser(description='SILC ensemble producer process.')
    parser.add_argument('--exchange', action='store_true')
    args = parser.parse_args()

    # get model and set into database
    model = create_torch_cnn()
    c = Client(False)
    c.set_model("torch_cnn", model, "TORCH")

    keyout = os.getenv("SSKEYOUT")
    keyin = os.getenv("SSKEYIN")

    assert keyout in ['producer_0', 'producer_1']

    if keyout == 'producer_0':
        c.set_data_source('producer_1' if args.exchange else 'producer_0')
        m = 1
    elif keyout == 'producer_1':
        c.set_data_source('producer_0' if args.exchange else 'producer_1')
        m = -10

    # setup input tensor
    data = torch.ones(1, 1, 3, 3).numpy()*m
    c.put_tensor("torch_cnn_input", data)


    input_exists = c.poll_entity("torch_cnn_input", EntityType.tensor, 100, 100)
    assert input_exists

    other_input = c.get_tensor("torch_cnn_input")

    # One process will send 1s, the other -10s. Sum is -9s.
    if args.exchange:
        assert np.sum(other_input+data)==-9.0*data.size
    else:
        assert np.all(other_input==data)

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    output_exists = c.poll_entity("torch_cnn_output", EntityType.tensor, 100, 100)
    assert output_exists

    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)
