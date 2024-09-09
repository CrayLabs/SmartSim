# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smartredis import Client


# simple MNIST in PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def create_torch_model():
    n = Net()
    example_forward_input = torch.rand(1, 1, 28, 28)
    module = torch.jit.trace(n, example_forward_input)
    model_buffer = io.BytesIO()
    torch.jit.save(module, model_buffer)
    return model_buffer.getvalue()


# random function from TorchScript API
def calc_svd(input_tensor):
    return input_tensor.svd()


def run(device, num_devices):
    # connect a client to the database
    client = Client(cluster=False)

    # test the SVD function
    tensor = np.random.randint(0, 100, size=(5, 3, 2)).astype(np.float32)
    client.put_tensor("input", tensor)
    client.set_function("svd", calc_svd)
    client.run_script("svd", "calc_svd", ["input"], ["U", "S", "V"])
    U = client.get_tensor("U")
    S = client.get_tensor("S")
    V = client.get_tensor("V")
    print(f"U: {U}, S: {S}, V: {V}")

    # test simple convNet
    net = create_torch_model()
    # 20 samples of "image" data
    example_forward_input = torch.rand(20, 1, 28, 28)
    client.put_tensor("input", example_forward_input.numpy())
    if device == "CPU":
        client.set_model("cnn", net, "TORCH", device=device)
        client.run_model("cnn", inputs=["input"], outputs=["output"])
    else:
        client.set_model_multigpu(
            "cnn", net, "TORCH", first_gpu=0, num_gpus=num_devices
        )
        client.run_model_multigpu(
            "cnn",
            offset=1,
            first_gpu=0,
            num_gpus=num_devices,
            inputs=["input"],
            outputs=["output"],
        )

    output = client.get_tensor("output")
    print(f"Prediction: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Torch test Script")
    parser.add_argument(
        "--device", type=str, default="CPU", help="device type for model execution"
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices to set the model on",
    )
    args = parser.parse_args()
    run(args.device, args.num_devices)
