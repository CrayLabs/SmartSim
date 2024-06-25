# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.utils import b64decode, b64encode
from dragon.globalservices.api_setup import connect_to_infrastructure
# isort: on
import argparse
import os


from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.infrastructure.worker.worker import TorchWorker
from smartsim._core.mli.infrastructure.control.workermanager import (
    WorkerManager,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Worker Manager")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()
    connect_to_infrastructure()
    ddict_str = os.environ["SS_DRG_DDICT"]
    ddict = DDict.attach(ddict_str)

    to_worker_channel = Channel.make_process_local()
    to_worker_manager_channel = Channel.make_process_local()
    channels = [Channel.make_process_local() for _ in range(100)]
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=to_worker_manager_channel, stream_channels=channels)
    ddict["to_worker_fli"] = b64encode(to_worker_fli.serialize())

    torch_worker = TorchWorker()

    worker_manager = WorkerManager(
        file_like_interface=to_worker_fli,
        worker=torch_worker,
        feature_store=None,
        as_service=True,
        cooldown=10,
        comm_channel_type=DragonCommChannel,
        device = args.device,
    )
    worker_manager.execute()
