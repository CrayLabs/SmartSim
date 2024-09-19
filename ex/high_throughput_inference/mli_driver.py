import argparse
import base64
import os
import shutil
import sys
import time
import typing as t

import cloudpickle

from smartsim import Experiment
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim.settings import DragonRunSettings
from smartsim.status import TERMINAL_STATUSES

parser = argparse.ArgumentParser("Mock application")
parser.add_argument("--log_max_batchsize", default=8, type=int)
parser.add_argument("--num_nodes_app", default=1, type=int)
args = parser.parse_args()

DEVICE = "gpu"
NUM_RANKS_PER_NODE = 16
NUM_NODES_APP = args.num_nodes_app
NUM_WORKERS = 1
BATCH_SIZE = 2
BATCH_TIMEOUT = 0.0
filedir = os.path.dirname(__file__)
worker_manager_script_name = os.path.join(filedir, "standalone_worker_manager.py")
app_script_name = os.path.join(filedir, "mock_app.py")
model_name = os.path.join(filedir, f"resnet50.{DEVICE}.pt")

transport: t.Literal["hsta", "tcp"] = "hsta"

os.environ["SMARTSIM_DRAGON_TRANSPORT"] = transport

exp_path = os.path.join(
    filedir,
    "benchmark",
    f"throughput_n{NUM_NODES_APP}_rpn{NUM_RANKS_PER_NODE}_timeout{BATCH_TIMEOUT}",
    f"samples{2**args.log_max_batchsize}",
)
try:
    shutil.rmtree(exp_path)
    time.sleep(2)
except:
    pass
os.makedirs(exp_path, exist_ok=True)
exp = Experiment("MLI_benchmark", launcher="dragon", exp_path=exp_path)

torch_worker_str = base64.b64encode(cloudpickle.dumps(TorchWorker)).decode("ascii")

worker_manager_rs: DragonRunSettings = exp.create_run_settings(
    sys.executable,
    [
        worker_manager_script_name,
        "--device",
        DEVICE,
        "--worker_class",
        torch_worker_str,
        "--batch_size",
        str(BATCH_SIZE),
        "--batch_timeout",
        str(BATCH_TIMEOUT),
        "--num_workers",
        str(NUM_WORKERS),
    ],
)

aff = []

worker_manager_rs.set_cpu_affinity(aff)
worker_manager_rs.set_gpu_affinity([0, 1, 2, 3])
worker_manager_rs.set_hostlist(["pinoak0037"])
worker_manager = exp.create_model("worker_manager", run_settings=worker_manager_rs)
worker_manager.attach_generator_files(to_copy=[worker_manager_script_name])

app_rs: DragonRunSettings = exp.create_run_settings(
    sys.executable,
    exe_args=[app_script_name, "--device", DEVICE, "--log_max_batchsize", str(args.log_max_batchsize)],
)
app_rs.set_tasks_per_node(NUM_RANKS_PER_NODE)
app_rs.set_nodes(NUM_NODES_APP)

app = exp.create_model("app", run_settings=app_rs)
app.attach_generator_files(to_copy=[app_script_name], to_symlink=[model_name])

exp.generate(worker_manager, app, overwrite=True)
exp.start(worker_manager, block=False)
exp.start(app, block=False)

while True:
    if exp.get_status(app)[0] in TERMINAL_STATUSES:
        time.sleep(10)
        exp.stop(worker_manager)
        break
    if exp.get_status(worker_manager)[0] in TERMINAL_STATUSES:
        time.sleep(10)
        exp.stop(app)
        break

print("Exiting.")
