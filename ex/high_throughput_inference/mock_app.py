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

# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
import dragon.channels
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.utils import b64decode, b64encode

# isort: on

import argparse
import io

import torch

from smartsim.log import get_logger

torch.set_num_interop_threads(16)
torch.set_num_threads(1)

logger = get_logger("App")
logger.info("Started app")

from collections import OrderedDict

from smartsim.log import get_logger, log_to_file
from smartsim._core.mli.client.protoclient import ProtoClient

logger = get_logger("App")




class ResNetWrapper:
    """Wrapper around a pre-rained ResNet model."""
    def __init__(self, name: str, model: str):
        """Initialize the instance.

        :param name: The name to use for the model
        :param model: The path to the pre-trained PyTorch model"""
        self._model = None  # torch.jit.load(model)
        self._name = name

        with open(model, "rb") as model_file:
            buffer = io.BytesIO(model_file.read())
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int = 32):
        """Create a random batch of data with the correct dimensions to
        invoke a ResNet model.

        :param batch_size: The desired number of samples to produce
        :returns: A PyTorch tensor"""
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self) -> bytes:
        """The content of a model file.

        :returns: The model bytes"""
        return self._serialized_model

    @property
    def name(self) -> str:
        """The name applied to the model.

        :returns: The name"""
        return self._name


def log(msg: str, rank: int) -> None:
    if rank == 0:
        logger.info(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device}.pt")

    client = ProtoClient(timing_on=True)
    if client._rank == 0:
        client.set_model(resnet.name, resnet.model)



    TOTAL_ITERATIONS = 100

    for log2_bsize in range(args.log_max_batchsize, args.log_max_batchsize + 1):
        b_size: int = 2**log2_bsize
        log(f"Batch size: {b_size}", client._rank)
        for iteration_number in range(TOTAL_ITERATIONS):
            sample_batch = resnet.get_batch(b_size)
            remote_result = client.run_model(resnet.name, sample_batch)
            logger.info(client.perf_timer.get_last("total_time"))


    client.perf_timer.print_timings(to_file=True, to_stdout=client._rank == 0)
