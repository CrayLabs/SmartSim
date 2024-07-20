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

import os
import sys
from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES
import time

DEVICE = "gpu"
filedir = os.path.dirname(__file__)
app_script_name = os.path.join(filedir, "mock_app_redis.py")
model_name = os.path.join(filedir, f"resnet50.{DEVICE.upper()}.pt")


exp_path = os.path.join(filedir, "redis_ai_multi")
os.makedirs(exp_path, exist_ok=True)
exp = Experiment("redis_ai_multi", launcher="slurm", exp_path=exp_path)

db = exp.create_database(interface="hsn0")

app_rs = exp.create_run_settings(
    sys.executable, exe_args = [app_script_name, "--device", DEVICE]
    )
app_rs.set_nodes(1)
app_rs.set_tasks(4)
app = exp.create_model("app", run_settings=app_rs)
app.attach_generator_files(to_copy=[app_script_name], to_symlink=[model_name])

exp.generate(db, app, overwrite=True)

exp.start(db, app, block=False)

while True:
    if exp.get_status(app)[0] in TERMINAL_STATUSES:
        exp.stop(db)
        break
    if exp.get_status(db)[0] in TERMINAL_STATUSES:
        exp.stop(app)
        break
    time.sleep(5)

print("Exiting.")