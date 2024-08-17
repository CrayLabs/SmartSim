import os
import base64
import cloudpickle
import sys
from smartsim import Experiment
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim.status import TERMINAL_STATUSES
from smartsim.settings import DragonRunSettings
import time
import typing as t

DEVICE = "gpu"
NUM_RANKS = 4
NUM_WORKERS = 1
filedir = os.path.dirname(__file__)
worker_manager_script_name = os.path.join(filedir, "standalone_workermanager.py")
app_script_name = os.path.join(filedir, "mock_app.py")
model_name = os.path.join(filedir, f"resnet50.{DEVICE}.pt")

transport: t.Literal["hsta", "tcp"] = "hsta"

os.environ["SMARTSIM_DRAGON_TRANSPORT"] = transport

exp_path = os.path.join(filedir, f"MLI_proto_{transport.upper()}")
os.makedirs(exp_path, exist_ok=True)
exp = Experiment("MLI_proto", launcher="dragon", exp_path=exp_path)

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
        str(NUM_RANKS//NUM_WORKERS),
        "--batch_timeout",
        str(0.00),
        "--num_workers",
        str(NUM_WORKERS)
    ],
)

aff = []

worker_manager_rs.set_cpu_affinity(aff)

worker_manager = exp.create_model("worker_manager", run_settings=worker_manager_rs)
worker_manager.attach_generator_files(to_copy=[worker_manager_script_name])

app_rs: DragonRunSettings = exp.create_run_settings(
    sys.executable,
    exe_args=[app_script_name, "--device", DEVICE, "--log_max_batchsize", str(6)],
)
app_rs.set_tasks_per_node(NUM_RANKS)


app = exp.create_model("app", run_settings=app_rs)
app.attach_generator_files(to_copy=[app_script_name], to_symlink=[model_name])

exp.generate(worker_manager, app, overwrite=True)
exp.start(worker_manager, app, block=False)

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
