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


import dragon

# pylint disable=import-error
import dragon.infrastructure.policy as dragon_policy
import dragon.infrastructure.process_desc as dragon_process_desc
import dragon.native.process as dragon_process
from dragon import fli
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.managed_memory import MemoryPool
from dragon.utils import b64decode, b64encode
# pylint enable=import-error

# isort: off
# isort: on

import argparse
import base64
import multiprocessing as mp
import os
import pickle
import socket
import sys
import time
import typing as t

import cloudpickle
import optparse
import os

from smartsim._core.entrypoints.service import Service
from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.control.requestdispatcher import (
    RequestDispatcher,
)
from smartsim._core.mli.infrastructure.control.workermanager import WorkerManager
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.worker.worker import MachineLearningWorkerBase

from smartsim.log import get_logger

logger = get_logger("Worker Manager Entry Point")

mp.set_start_method("dragon")

pid = os.getpid()
affinity = os.sched_getaffinity(pid)
logger.info(f"Entry point: {socket.gethostname()}, {affinity}")
logger.info(f"CPUS: {os.cpu_count()}")



def service_as_dragon_proc(
    service: Service, cpu_affinity: list[int], gpu_affinity: list[int]
) -> dragon_process.Process:

    options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
    local_policy = dragon_policy.Policy(
        placement=dragon_policy.Policy.Placement.HOST_NAME,
        host_name=socket.gethostname(),
        cpu_affinity=cpu_affinity,
        gpu_affinity=gpu_affinity,
    )
    return dragon_process.Process(
        target=service.execute,
        args=[],
        cwd=os.getcwd(),
        policy=local_policy,
        options=options,
        stderr=dragon_process.Popen.STDOUT,
        stdout=dragon_process.Popen.STDOUT,
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Worker Manager")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices="gpu cpu".split(),
        help="Device on which the inference takes place",
    )
    parser.add_argument(
        "--worker_class",
        type=str,
        required=True,
        help="Serialized class of worker to run",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to run"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="How many requests the workers will try to aggregate before processing them",
    )
    parser.add_argument(
        "--batch_timeout",
        type=float,
        default=0.001,
        help="How much time (in seconds) should be waited before processing an incomplete aggregated request",
    )
    args = parser.parse_args()

    connect_to_infrastructure()
    ddict_str = os.environ["_SMARTSIM_INFRA_BACKBONE"]
    ddict = DDict.attach(ddict_str)

    to_worker_channel = Channel.make_process_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    to_worker_fli_serialized = to_worker_fli.serialize()
    ddict["to_worker_fli"] = to_worker_fli_serialized

    arg_worker_type = cloudpickle.loads(
        base64.b64decode(args.worker_class.encode("ascii"))
    )

    dfs = DragonFeatureStore(ddict)
    comm_channel = DragonFLIChannel(to_worker_fli_serialized)

    descriptor = base64.b64encode(to_worker_fli_serialized).decode("utf-8")
    os.environ["_SMARTSIM_REQUEST_QUEUE"] = descriptor

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    dispatcher = RequestDispatcher(
        batch_timeout=args.batch_timeout,
        batch_size=args.batch_size,
        config_loader=config_loader,
        worker_type=arg_worker_type,
    )

    wms = []
    worker_device = args.device
    for wm_idx in range(args.num_workers):

        worker_manager =  WorkerManager(
            config_loader=config_loader,
            worker_type=arg_worker_type,
            as_service=True,
            cooldown=10,
            device=worker_device,
            dispatcher_queue=dispatcher.task_queue,
        )

        wms.append(worker_manager)

    wm_affinity: list[int] = []
    disp_affinity: list[int] = []

    # This is hardcoded for a specific type of node:
    # the GPU-to-CPU mapping is taken from the nvidia-smi tool
    # TODO can this be computed on the fly?
    gpu_to_cpu_aff: dict[int, list[int]] = {}
    gpu_to_cpu_aff[0] = list(range(48,64)) + list(range(112,128))
    gpu_to_cpu_aff[1] = list(range(32,48)) + list(range(96,112))
    gpu_to_cpu_aff[2] = list(range(16,32)) + list(range(80,96))
    gpu_to_cpu_aff[3] = list(range(0,16)) + list(range(64,80))

    worker_manager_procs = []
    for worker_idx in range(args.num_workers):
        wm_cpus = len(gpu_to_cpu_aff[worker_idx]) - 4
        wm_affinity = gpu_to_cpu_aff[worker_idx][:wm_cpus]
        disp_affinity.extend(gpu_to_cpu_aff[worker_idx][wm_cpus:])
        worker_manager_procs.append(service_as_dragon_proc(
                worker_manager, cpu_affinity=wm_affinity, gpu_affinity=[worker_idx]
            ))

    dispatcher_proc = service_as_dragon_proc(dispatcher, cpu_affinity=disp_affinity, gpu_affinity=[])

    # TODO: use ProcessGroup and restart=True?
    all_procs = [dispatcher_proc, *worker_manager_procs]

    print(f"Dispatcher proc: {dispatcher_proc}")
    for proc in all_procs:
        proc.start()

    while all(proc.is_alive for proc in all_procs):
        time.sleep(1)
