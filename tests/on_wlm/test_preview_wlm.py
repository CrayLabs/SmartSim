# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

from os import path as osp

import numpy as np
import pytest
from jinja2.filters import FILTERS

from smartsim import Experiment
from smartsim._core import Manifest, previewrenderer
from smartsim._core.config import CONFIG
from smartsim.database import Orchestrator
from smartsim.settings import QsubBatchSettings, RunSettings

pytestmark = pytest.mark.slow_tests

on_wlm = (pytest.test_launcher in pytest.wlm_options,)


@pytest.fixture
def choose_host():
    def _choose_host(wlmutils, index: int = 0):
        hosts = wlmutils.get_test_hostlist()
        if hosts:
            return hosts[index]
        return None

    return _choose_host


def add_batch_resources(wlmutils, batch_settings):
    if isinstance(batch_settings, QsubBatchSettings):
        for key, value in wlmutils.get_batch_resources().items():
            batch_settings.set_resource(key, value)


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_wlm_run_commands_cluster_orc_model(
    test_dir, coloutils, fileutils, wlmutils
):
    """
    Test preview of wlm run command and run aruguments on a
    orchestrator and model
    """

    exp_name = "test-preview-orc-model"
    launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()
    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
        db_identifier="testdb_reg",
    )

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    # Create model with colocated database
    smartsim_model = coloutils.setup_test_colo(
        fileutils, "uds", exp, test_script, db_args, on_wlm=on_wlm
    )

    preview_manifest = Manifest(orc, smartsim_model)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    if pytest.test_launcher != "dragon":
        assert "Run Command" in output
        assert "ntasks" in output
    assert "Run Arguments" in output
    assert "nodes" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_model_on_wlm(fileutils, test_dir, wlmutils):
    """
    Test preview of wlm run command and run aruguments for a model
    """
    exp_name = "test-preview-model-wlm"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings1 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    settings2 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)

    preview_manifest = Manifest(M1, M2)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    if pytest.test_launcher != "dragon":
        assert "Run Command" in output
        assert "ntasks" in output
        assert "time" in output
        assert "nodes" in output
    assert "Run Arguments" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_batch_model(fileutils, test_dir, wlmutils):
    """Test the preview of a model with batch settings"""

    exp_name = "test-batch-model"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    batch_settings = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch_settings.set_account(wlmutils.get_test_account())
    add_batch_resources(wlmutils, batch_settings)
    run_settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    model = exp.create_model(
        "model", path=test_dir, run_settings=run_settings, batch_settings=batch_settings
    )
    model.set_path(test_dir)

    preview_manifest = Manifest(model)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    assert "Batch Launch: True" in output
    assert "Batch Command" in output
    assert "Batch Arguments" in output
    assert "nodes" in output
    assert "time" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_batch_ensemble(fileutils, test_dir, wlmutils):
    """Test preview of a batch ensemble"""

    exp_name = "test-preview-batch-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")
    add_batch_resources(wlmutils, batch)

    batch.set_account(wlmutils.get_test_account())
    ensemble = exp.create_ensemble("batch-ens", batch_settings=batch)
    ensemble.add_model(M1)
    ensemble.add_model(M2)
    ensemble.set_path(test_dir)

    preview_manifest = Manifest(ensemble)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    assert "Batch Launch: True" in output
    assert "Batch Command" in output
    assert "Batch Arguments" in output
    assert "nodes" in output
    assert "time" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_launch_command(test_dir, wlmutils, choose_host):
    """Test preview launch command for orchestrator, models, and
    ensembles"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_preview_launch_command"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # create regular database
    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
    )

    model_params = {"port": 6379, "password": "unbreakable_password"}
    rs1 = RunSettings("bash", "multi_tags_template.sh")
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model(
        "echo-hello", run_settings=rs1, params=model_params
    )

    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    # setup ensemble parameter space
    learning_rate = list(np.linspace(0.01, 0.5))
    train_params = {"LR": learning_rate}

    run = exp.create_run_settings(exe="python", exe_args="./train-model.py")

    ensemble = exp.create_ensemble(
        "Training-Ensemble",
        params=train_params,
        params_as_args=["LR"],
        run_settings=run,
        perm_strategy="random",
        n_models=4,
    )

    preview_manifest = Manifest(orc, spam_eggs_model, hello_world_model, ensemble)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    assert "orchestrator" in output
    assert "echo-spam" in output
    assert "echo-hello" in output

    assert "Training-Ensemble" in output
    assert "me: Training-Ensemble_0" in output
    assert "Training-Ensemble_1" in output
    assert "Training-Ensemble_2" in output
    assert "Training-Ensemble_3" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_batch_launch_command(fileutils, test_dir, wlmutils):
    """Test the preview of a model with batch settings"""

    exp_name = "test-batch-entities"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    batch_settings = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch_settings.set_account(wlmutils.get_test_account())
    add_batch_resources(wlmutils, batch_settings)
    run_settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    model = exp.create_model(
        "model", path=test_dir, run_settings=run_settings, batch_settings=batch_settings
    )
    model.set_path(test_dir)

    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    orc.set_batch_arg("account", "ACCOUNT")

    preview_manifest = Manifest(orc, model)
    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Batch Launch: True" in output
    assert "Batch Command" in output
    assert "Batch Arguments" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_ensemble_batch(test_dir, wlmutils):
    """
    Test preview of client configuration and key prefixing in Ensemble preview
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(
        "test-preview-ensemble-clientconfig", exp_path=test_dir, launcher=test_launcher
    )
    # Create Orchestrator
    db = exp.create_database(port=6780, interface="lo")
    exp.generate(db, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    # Create ensemble
    batch_settings = exp.create_batch_settings(nodes=1, time="00:01:00")
    batch_settings.set_account(wlmutils.get_test_account())
    add_batch_resources(wlmutils, batch_settings)
    ensemble = exp.create_ensemble(
        "fd_simulation", run_settings=rs1, batch_settings=batch_settings, replicas=2
    )
    # enable key prefixing on ensemble
    ensemble.enable_key_prefixing()
    exp.generate(ensemble, overwrite=True)
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])
    # Create model
    ml_model = exp.create_model("tf_training", rs2)

    for sim in ensemble.entities:
        ml_model.register_incoming_entity(sim)

    exp.generate(ml_model, overwrite=True)

    preview_manifest = Manifest(db, ml_model, ensemble)

    # Call preview renderer for testing output
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Client Configuration" in output
    assert "Database Identifier" in output
    assert "Database Backend" in output
    assert "Type" in output


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_preview_ensemble_db_script(wlmutils, test_dir):
    """
    Test preview of a torch script on a model in an ensemble.
    """
    # Initialize the Experiment and set the launcher to auto
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment("getting-started", launcher=test_launcher)

    orch = exp.create_database(db_identifier="test_db1")
    orch_2 = exp.create_database(db_identifier="test_db2", db_nodes=3)
    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe="python", exe_args="params.py")
    model_settings_2 = exp.create_run_settings(exe="python", exe_args="params.py")
    model_settings_3 = exp.create_run_settings(exe="python", exe_args="params.py")
    # Initialize a Model object
    model_instance = exp.create_model("model_name", model_settings)
    model_instance_2 = exp.create_model("model_name_2", model_settings_2)
    batch = exp.create_batch_settings(time="24:00:00", account="test")
    ensemble = exp.create_ensemble(
        "ensemble", batch_settings=batch, run_settings=model_settings_3, replicas=2
    )
    ensemble.add_model(model_instance)
    ensemble.add_model(model_instance_2)

    # TorchScript string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Attach TorchScript to Model
    model_instance.add_script(
        name="example_script",
        script=torch_script_str,
        device="GPU",
        devices_per_node=2,
        first_device=0,
    )
    preview_manifest = Manifest(ensemble, orch, orch_2)

    # Call preview renderer for testing output
    output = previewrenderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Torch Script" in output
