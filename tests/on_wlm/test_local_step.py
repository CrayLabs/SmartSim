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
import uuid

import pytest

from smartsim import Experiment, status
from smartsim.settings import RunSettings

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Test is only for Slurm WLM systems")

"""
Test execution of local steps within the WLM
"""


def test_local_env_pass_implicit(fileutils, test_dir) -> None:
    """Ensure implicitly exported env is available to running task"""
    exp_value = str(uuid.uuid4())
    env_key = "test_local_env_pass_implicit"
    os.environ[env_key] = exp_value

    exp_dir = f"{test_dir}/exp"
    os.makedirs(exp_dir)
    script = fileutils.get_test_conf_path("check_env.py")

    exp = Experiment("LRZ", exp_path=exp_dir, launcher="slurm")

    exe_name = "python"
    exe_args = [script, env_key]

    # Create the RunSettings associated with the workload manager (WLM) run command
    run_args = {"--nodes": 1, "--ntasks": 1, "--time": "00:01:00"}
    # NOTE: not passing env_args into run_settings here, relying on --export=ALL default
    settings = RunSettings(exe_name, exe_args, run_command="srun", run_args=run_args)
    app_name = "echo_app"
    app = exp.create_model(app_name, settings)

    # generate the experiment structure and start the model
    exp.generate(app, overwrite=True)
    exp.start(app, block=True, summary=False)

    assert env_key not in settings.env_vars
    os.environ.pop(env_key)

    with open(f"{exp_dir}/{app_name}/{app_name}.out") as app_outfile:
        app_output = app_outfile.read()

    # verify application was able to access the env var
    assert f"{env_key}=={exp_value}" in app_output


def test_local_env_pass_explicit(fileutils, test_dir) -> None:
    """Ensure explicitly exported env is available to running task"""
    exp_value = str(uuid.uuid4())
    env_key = "test_local_env_pass_explicit"

    assert env_key not in os.environ

    script = fileutils.get_test_conf_path("check_env.py")

    exp_dir = f"{test_dir}/exp"
    os.makedirs(exp_dir)
    exp = Experiment("LRZ", exp_path=exp_dir, launcher="slurm")

    exe_name = "python"
    exe_args = [script, env_key]

    # Create the RunSettings associated with the workload manager (WLM) run command
    run_args = {"--nodes": 1, "--ntasks": 1, "--time": "00:01:00"}
    env_vars = {env_key: exp_value}  # <-- explicitly passing a new env var to task
    settings = RunSettings(
        exe_name, exe_args, run_command="srun", run_args=run_args, env_vars=env_vars
    )
    app_name = "echo_app"
    app = exp.create_model(app_name, settings)

    # generate the experiment structure and start the model
    exp.generate(app, overwrite=True)
    exp.start(app, block=True, summary=False)

    assert env_key in settings.env_vars

    with open(f"{exp_dir}/{app_name}/{app_name}.out") as app_outfile:
        app_output = app_outfile.read()

    # verify application was able to access the env var
    assert f"{env_key}=={exp_value}" in app_output
