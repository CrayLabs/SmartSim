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

import argparse
import io
import numpy
import time
import torch
from smartsim.log import get_logger
from smartredis import Client

logger = get_logger("App")

class ResNetWrapper():
    def __init__(self, name: str, model: str):
        self._model = torch.jit.load(model)
        self._name = name
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._model, self.get_batch())
        torch.jit.save(scripted, buffer)
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int=32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

    @property
    def name(self):
        return self._name

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device.upper()}.pt")

    client = Client(cluster=False, address=None)
    client.set_model(resnet.name, resnet.model, backend='TORCH', device=args.device.upper())

    total_iterations = 100
    timings=[]
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        logger.info(f"Batch size: {batch_size}")
        for iteration_number in range(total_iterations + int(batch_size==1)):
            timing = [batch_size]
            logger.info(f"Iteration: {iteration_number}")
            start = time.perf_counter()
            client.put_tensor(name="batch", data=resnet.get_batch(batch_size).numpy())
            client.run_model(name=resnet.name, inputs=["batch"], outputs=["result"])
            result = client.get_tensor(name="result")
            end = time.perf_counter()
            timing.append(end-start)
            timings.append(timing)



    timings_np = numpy.asarray(timings)
    numpy.save("timings.npy", timings_np)
    for timing in timings:
        print(" ".join(str(t) for t in timing))
