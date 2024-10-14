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

from mpi4py import MPI
import numpy
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

from smartsim.log import get_logger
from smartsim._core.mli.client.protoclient import ProtoClient

logger = get_logger("App")


class ResNetWrapper:
    def __init__(
        self,
        name: str,
        model: tf.keras.Model,
    ):
        self._get_tf_model(model)
        self._name = name

    def _get_tf_model(self, model: tf.keras.Model):
        real_model = tf.function(model).get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        _, graph_def = convert_variables_to_constants_v2_as_graph(real_model)
        self._serialized_model = graph_def.SerializeToString()

    # pylint: disable-next=no-self-use
    def get_batch(self, batch_size: int = 32):
        """Create a random batch of data with the correct dimensions to
        invoke a ResNet model.

        :param batch_size: The desired number of samples to produce
        :returns: A PyTorch tensor"""
        return numpy.random.randn(batch_size, 224, 224, 3).astype(numpy.float32)

    @property
    def model(self):
        """The content of a model file.

        :returns: The model bytes"""
        return self._serialized_model

    @property
    def name(self):
        """The name applied to the model.

        :returns: The name"""
        return self._name


def log(msg: str, rank_: int) -> None:
    if rank_ == 0:
        logger.info(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", tf.keras.applications.ResNet50())

    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    client = ProtoClient(timing_on=True, rank=rank)

    if rank == 0:
        client.set_model(resnet.name, resnet.model)

    comm_world.Barrier()

    TOTAL_ITERATIONS = 100

    for log2_bsize in range(args.log_max_batchsize, args.log_max_batchsize + 1):
        b_size: int = 2**log2_bsize
        log(f"Batch size: {b_size}", rank)
        for iteration_number in range(TOTAL_ITERATIONS):
            sample_batch = resnet.get_batch(b_size)
            remote_result = client.run_model(resnet.name, sample_batch)
            log(
                f"Completed iteration: {iteration_number} in "
                f"{client.perf_timer.get_last('total_time')} seconds",
                rank,
            )

    client.perf_timer.print_timings(to_file=True, to_stdout=rank == 0)
