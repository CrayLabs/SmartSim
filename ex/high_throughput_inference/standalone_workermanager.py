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
import dragon.globalservices.pool as dragon_gs_pool
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

from smartsim._core.entrypoints.service import Service
from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.control.requestdispatcher import (
    RequestDispatcher,
)
from smartsim._core.mli.infrastructure.control.workermanager import WorkerManager
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.worker.worker import MachineLearningWorkerBase

mp.set_start_method("dragon")

pid = os.getpid()
affinity = os.sched_getaffinity(pid)
print("Entry point:", socket.gethostname(), affinity)
print("CPUS:", os.cpu_count())


def create_request_dispatcher(
    batch_size: int,
    batch_timeout: float,
    comm_channel_type: t.Type[CommChannelBase],
    worker_type: t.Type[MachineLearningWorkerBase],
    config_loader: EnvironmentConfigLoader,
) -> RequestDispatcher:
    mem_pool = MemoryPool.attach(dragon_gs_pool.create(2 * 1024**3).sdesc)

    return RequestDispatcher(
        batch_timeout=batch_timeout,
        batch_size=batch_size,
        config_loader=config_loader,
        comm_channel_type=comm_channel_type,
        mem_pool=mem_pool,
        worker_type=worker_type,
    )


def create_worker_manager(
    worker_type: t.Type[MachineLearningWorkerBase],
    config_loader: EnvironmentConfigLoader,
    device: str,
    dispatcher_queue: mp.Queue,
) -> WorkerManager:
    return WorkerManager(
        config_loader=config_loader,
        worker_type=worker_type,
        as_service=True,
        cooldown=10,
        comm_channel_type=DragonCommChannel,
        device=device,
        task_queue=dispatcher_queue,
    )


def service_as_dragon_proc(
    service: Service, cpu_affinity: list[int], gpu_affinity: list[int]
) -> dragon_process.Process:

    options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
    local_policy = dragon_policy.Policy(
        placement=dragon_policy.Policy.Placement.HOST_NAME,
        host_name=socket.gethostname(),
        affinity=dragon_policy.Policy.Affinity.SPECIFIC,
        cpu_affinity=cpu_affinity,
        gpu_affinity=gpu_affinity,
    )
    proc = dragon_process.Process(
        target=service.execute,
        args=[],
        cwd=os.getcwd(),
        policy=local_policy,
        options=options,
        stderr=dragon_process.Popen.PIPE,
        stdout=dragon_process.Popen.STDOUT,
    )

    return proc


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
    ddict_str = os.environ["SS_DRG_DDICT"]
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

    os.environ["SSFeatureStore"] = base64.b64encode(pickle.dumps(dfs)).decode("utf-8")
    os.environ["SSQueue"] = base64.b64encode(to_worker_fli_serialized).decode("utf-8")

    ss_config_loader = EnvironmentConfigLoader()

    dispatcher = create_request_dispatcher(
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        comm_channel_type=DragonCommChannel,
        worker_type=arg_worker_type,
        config_loader=ss_config_loader,
    )

    worker_manager = create_worker_manager(
        worker_type=arg_worker_type,
        config_loader=ss_config_loader,
        device=args.device,
        dispatcher_queue=dispatcher.task_queue,
    )

    wm_affinity: list[int] = []
    disp_affinity: list[int] = []
    if sys.platform != "darwin":
        curr_affinity: list[int] = list(os.sched_getaffinity(os.getpid()))
        wm_cpus = 3 * len(curr_affinity) // 4
        disp_affinity = curr_affinity[wm_cpus:]
        wm_affinity = curr_affinity[:wm_cpus]

    dispatcher_proc = service_as_dragon_proc(dispatcher, cpu_affinity=disp_affinity, gpu_affinity=[])
    worker_manager_proc = service_as_dragon_proc(
        worker_manager, cpu_affinity=wm_affinity, gpu_affinity=[]
    )

    dispatcher_proc.start()
    worker_manager_proc.start()

    while all(proc.is_alive for proc in [dispatcher_proc, worker_manager_proc]):
        time.sleep(1)
