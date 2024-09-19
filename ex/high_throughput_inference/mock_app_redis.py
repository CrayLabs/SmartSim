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
from mpi4py import MPI
from smartsim.log import get_logger
from smartsim._core.utils.timings import PerfTimer
from smartredis import Client

logger = get_logger("App")

class ResNetWrapper():
    def __init__(self, name: str, model: str):
        self._model = None
        self._name = name
        with open(model, "rb") as model_file:
            buffer = io.BytesIO(model_file.read())
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int=32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

    @property
    def name(self):
        return self._name


def log(msg: str, rank: int) -> None:
    if rank == 0:
        logger.info(msg)

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device}.pt")

    client = Client(cluster=False, address=None)

    if rank == 0:
        client.set_model(resnet.name, resnet.model, backend='TORCH', device=args.device.upper())

    comm.Barrier()

    perf_timer: PerfTimer = PerfTimer(debug=False, timing_on=True, prefix=f"redis{rank}_")

    total_iterations = 100
    timings=[]
    for log2_bsize in range(args.log_max_batchsize, args.log_max_batchsize+1):
        batch_size: int = 2**log2_bsize
        log(f"Batch size: {batch_size}", rank)
        for iteration_number in range(total_iterations):
            perf_timer.start_timings("batch_size", batch_size)
            input_name = f"batch_{rank}"
            output_name = f"result_{rank}"
            client.put_tensor(name=input_name, data=resnet.get_batch(batch_size).numpy())
            perf_timer.measure_time("send_request")
            client.run_model(name=resnet.name, inputs=[input_name], outputs=[output_name])
            perf_timer.measure_time("run_model")
            result = client.get_tensor(name=output_name)
            perf_timer.measure_time("receive_response")
            perf_timer.end_timings()
            comm.Barrier()
            log(f"Completed iteration: {iteration_number} in {perf_timer.get_last('total_time')} seconds", rank)


    perf_timer.print_timings(True, to_stdout=rank==0)
