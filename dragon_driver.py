from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES
import os
import time

WAIT = False

def wait_for_input(msg: str):
    if WAIT:
        input(msg)

os.makedirs("./dragon_exp", exist_ok=True)

exp = Experiment("DRAGON", exp_path="./dragon_exp", launcher="dragon")

rs = exp.create_run_settings("sleep", exe_args=["25"])
model = exp.create_model("sleep_model", run_settings = rs)

exp.generate(model, overwrite=True)

wait_for_input("\nModel directory tree generated. Press any key to start model execution...\n")

exp.start(model, block=False)


rs_ens = exp.create_run_settings("hostname")
ensemble = exp.create_ensemble("ensemble", run_settings=rs_ens, replicas = 10)

exp.generate(ensemble, overwrite=True)

wait_for_input("\nEnsemble directory tree generated. Press any key to start ensemble execution...\n")

exp.start(ensemble, block=True)

while not exp.get_status(model)[0] in TERMINAL_STATUSES:
    print (">>>>", exp.get_status(model)[0])

wait_for_input("\nStart a model and stop it after 5 seconds...\n")
exp.start(model, block=False)
time.sleep(5)
exp.stop(model)


wait_for_input("\nStart clustered DB and stop it after 5 seconds...\n")
orc = exp.create_database(db_nodes=3)
exp.generate(orc)

exp.start(orc)
time.sleep(5)
exp.stop(orc)