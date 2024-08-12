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
from dragon.data.ddict.ddict import DDict
from dragon.utils import b64decode, b64encode
from dragon.globalservices.api_setup import connect_to_infrastructure

# isort: on
import argparse
import base64
import cloudpickle
import optparse
import os

from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.control.workermanager import WorkerManager
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader


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

    args = parser.parse_args()
    connect_to_infrastructure()
    ddict_str = os.environ["_SMARTSIM_INFRA_BACKBONE"]
    ddict = DDict.attach(ddict_str)

    to_worker_channel = Channel.make_process_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    to_worker_fli_serialized = to_worker_fli.serialize()
    ddict["to_worker_fli"] = to_worker_fli_serialized

    worker_type_name = base64.b64decode(args.worker_class.encode("ascii"))
    torch_worker = cloudpickle.loads(worker_type_name)()

    descriptor = base64.b64encode(to_worker_fli_serialized).decode("utf-8")
    os.environ["_SMARTSIM_REQUEST_QUEUE"] = descriptor

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    worker_manager = WorkerManager(
        config_loader=config_loader,
        worker=torch_worker,
        as_service=True,
        cooldown=10,
        device=args.device,
    )
    worker_manager.execute()
