# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    input_exists = c.poll_tensor("torch_cnn_input", 100, 1000)
    assert input_exists

    other_input = c.get_tensor("torch_cnn_input")

    if args.exchange:
        assert np.all(other_input == data_other)
    else:
        assert np.all(other_input == data)

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    output_exists = c.poll_tensor("torch_cnn_output", 100, 1000)
    assert output_exists

    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)
