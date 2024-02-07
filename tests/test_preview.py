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
import pathlib
import sys
from os import path as osp

import numpy as np
import pytest

from smartsim import Experiment
from smartsim._core import Manifest, previewrenderer
from smartsim.error.errors import PreviewFormatError
from smartsim.settings import RunSettings


@pytest.fixture
def choose_host():
    def _choose_host(wlmutils, index: int = 0):
        hosts = wlmutils.get_test_hostlist()
        if hosts:
            return hosts[index]
        return None

    return _choose_host


def test_experiment_preview(test_dir, wlmutils):
    """Test correct preview output fields for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = previewrenderer.render(exp)

    # Evaluate output
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert set(["Experiment", "Experiment Path", "Launcher"]).issubset(summary_dict)


def test_experiment_preview_properties(test_dir, wlmutils):
    """Test correct preview output properties for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = previewrenderer.render(exp)

    # Evaluate output
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert exp.name == summary_dict["Experiment"]
    assert exp.exp_path == summary_dict["Experiment Path"]
    assert exp.launcher == summary_dict["Launcher"]


def test_preview_output_format_html_to_file(test_dir, wlmutils):
    """Test that an html file is rendered for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_preview_output_format_html"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    filename = "test_preview_output_format_html.html"
    path = pathlib.Path(test_dir) / filename

    # Execute preview method
    exp.preview(output_format="html", output_filename=str(path))

    # Evaluate output
    assert path.exists()
    assert path.is_file()


def test_preview_model_output_format(test_dir, wlmutils):
    """
    Test that an html file is rendered for Model preview
    """
    # Prepare entities
    exp_name = "test_model_preview-html"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    model_params = {"port": 6379, "password": "unbreakable_password"}
    rs1 = RunSettings("bash", "multi_tags_template.sh")
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model(
        "echo-hello", run_settings=rs1, params=model_params
    )
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    filename = "test_model_preview_output_format.html"
    path = pathlib.Path(test_dir) / filename

    # Execute preview method
    exp.preview(
        hello_world_model,
        spam_eggs_model,
        output_format="html",
        output_filename=str(path),
    )

    # Evaluate output
    assert path.exists()
    assert path.is_file()


def test_model_preview(test_dir, wlmutils):
    """
    Test correct preview output fields for Model preview
    """
    # Prepare entities
    exp_name = "test_model_preview"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    model_params = {"port": 6379, "password": "unbreakable_password"}
    rs1 = RunSettings("bash", "multi_tags_template.sh")
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model(
        "echo-hello", run_settings=rs1, params=model_params
    )
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)

    # Execute preview method
    rendered_preview = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Model name" in rendered_preview
    assert "Executable" in rendered_preview
    assert "Executable Arguments" in rendered_preview
    assert "Batch Launch" in rendered_preview
    assert "Model parameters" in rendered_preview


def test_model_preview_properties(test_dir, wlmutils):
    """
    Test correct preview output properties for Model preview
    """
    # Prepare entities
    exp_name = "test_model_preview_parameters"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    model_params = {"port": 6379, "password": "unbreakable_password"}
    rs1 = RunSettings("bash", "multi_tags_template.sh")

    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model(
        "echo-hello", run_settings=rs1, params=model_params
    )

    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)
    preview_manifest = Manifest(hello_world_model, spam_eggs_model)

    # Execute preview method
    rendered_preview = previewrenderer.render(exp, preview_manifest)

    # Evaluate output for hello world model
    assert "echo-hello" in rendered_preview
    assert "/bin/bash" in rendered_preview
    assert "multi_tags_template.sh" in rendered_preview
    assert "False" in rendered_preview
    assert "port" in rendered_preview
    assert "password" in rendered_preview
    assert "6379" in rendered_preview
    assert "unbreakable_password" in rendered_preview

    assert "echo-hello" == hello_world_model.name
    assert "/bin/bash" in hello_world_model.run_settings.exe[0]
    assert "multi_tags_template.sh" == hello_world_model.run_settings.exe_args[0]
    assert None == hello_world_model.batch_settings
    assert "port" in list(hello_world_model.params.items())[0]
    assert 6379 in list(hello_world_model.params.items())[0]
    assert "password" in list(hello_world_model.params.items())[1]
    assert "unbreakable_password" in list(hello_world_model.params.items())[1]

    # Evaluate outputfor spam eggs model
    assert "echo-spam" in rendered_preview
    assert "/bin/echo" in rendered_preview
    assert "spam" in rendered_preview
    assert "eggs" in rendered_preview
    assert "echo-spam" == spam_eggs_model.name
    assert "/bin/echo" in spam_eggs_model.run_settings.exe[0]
    assert "spam" == spam_eggs_model.run_settings.exe_args[0]
    assert "eggs" == spam_eggs_model.run_settings.exe_args[1]


def test_model_with_tagged_files(fileutils, test_dir, wlmutils):
    """
    Test model with tagged files in preview.
    """
    # Prepare entities
    exp_name = "test_model_preview_parameters"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    model_params = {"port": 6379, "password": "unbreakable_password"}
    model_settings = RunSettings("bash", "multi_tags_template.sh")

    hello_world_model = exp.create_model(
        "echo-hello", run_settings=model_settings, params=model_params
    )

    config = fileutils.get_test_conf_path(
        osp.join("generator_files", "multi_tags_template.sh")
    )
    hello_world_model.attach_generator_files(to_configure=[config])
    exp.generate(hello_world_model, overwrite=True)

    preview_manifest = Manifest(hello_world_model)

    # Execute preview method
    rendered_preview = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Tagged Files for model configuration" in rendered_preview
    assert "generator_files/multi_tags_template.sh" in rendered_preview
    assert "generator_files/multi_tags_template.sh" in hello_world_model.files.tagged[0]


def test_model_key_prefixing(test_dir, wlmutils):
    """
    Test preview for enabling key prefixing for a Model
    """

    # Prepare entities
    exp_name = "test_model_key_prefixing"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    db = exp.create_database(port=6780, interface="lo")
    exp.generate(db, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    model = exp.create_model("model_test", run_settings=rs1)
    # enable key prefixing on ensemble
    model.enable_key_prefixing()
    exp.generate(model, overwrite=True)

    preview_manifest = Manifest(db, model)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Key prefix" in output
    assert "model_test" in output


def test_ensembles_preview(test_dir, wlmutils):
    """
    Test ensemble preview fields are correct in template render
    """
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(
        "test-ensembles-preview", exp_path=test_dir, launcher=test_launcher
    )

    # setup ensemble parameter space
    learning_rate = list(np.linspace(0.01, 0.5))
    train_params = {"LR": learning_rate}

    # define how each member should run
    run = exp.create_run_settings(exe="python", exe_args="./train-model.py")

    ensemble = exp.create_ensemble(
        "Training-Ensemble",
        params=train_params,
        params_as_args=["LR"],
        run_settings=run,
        perm_strategy="random",
        n_models=4,
    )

    preview_manifest = Manifest(ensemble)
    output = previewrenderer.render(exp, preview_manifest)

    assert "Ensemble name" in output
    assert "Members" in output
    assert "Ensemble parameters" in output


def test_preview_models_and_ensembles(test_dir, wlmutils):
    """
    Test preview of separate model entity and ensemble entity
    """
    exp_name = "test-model-and-ensemble"
    test_dir = pathlib.Path(test_dir) / exp_name
    test_dir.mkdir(parents=True)
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher=test_launcher)

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)
    hello_ensemble = exp.create_ensemble("echo-ensemble", run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model, hello_ensemble)
    output = previewrenderer.render(exp, preview_manifest)

    assert "Models" in output
    assert "echo-hello" in output
    assert "echo-spam" in output

    assert "Ensembles" in output
    assert "echo-ensemble_1" in output
    assert "echo-ensemble_2" in output


def test_ensemble_preview_client_configuration(test_dir, wlmutils):
    """
    Test client configuration and key prefixing in Ensemble preview
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment("key_prefix_test", exp_path=test_dir, launcher=test_launcher)
    # Create Orchestrator
    db = exp.create_database(port=6780, interface="lo")
    exp.generate(db, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    # Create ensemble
    ensemble = exp.create_ensemble("fd_simulation", run_settings=rs1, replicas=2)
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
    output = previewrenderer.render(exp, preview_manifest)

    assert "Client Configuration" in output
    assert "Database identifier" in output
    assert "Database backend" in output
    assert "Type" in output
    assert "Outgoing key collision prevention (key prefixing)" in output
    assert "Tensors: On" in output
    assert "DataSets: On" in output
    assert "Models/Scripts: Off" in output
    assert "Aggregation Lists: On" in output


def test_ensemble_preview_attached_files(fileutils, test_dir, wlmutils):
    """
    Test the preview of tagged, copy, and symlink files attached
    to an ensemble
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment("attached-files-test", exp_path=test_dir, launcher=test_launcher)
    ensemble = exp.create_ensemble(
        "dir_test", replicas=1, run_settings=RunSettings("python", exe_args="sleep.py")
    )
    ensemble.entities = []
    params = {"THERMO": [10, 20], "STEPS": [20, 30]}
    ensemble = exp.create_ensemble(
        "dir_test",
        params=params,
        run_settings=RunSettings("python", exe_args="sleep.py"),
    )
    gen_dir = fileutils.get_test_conf_path(osp.join("generator_files", "test_dir"))
    symlink_dir = fileutils.get_test_conf_path(
        osp.join("generator_files", "to_symlink_dir")
    )
    copy_dir = fileutils.get_test_conf_path(osp.join("generator_files", "to_copy_dir"))

    ensemble.attach_generator_files()
    ensemble.attach_generator_files(
        to_configure=[gen_dir, copy_dir], to_copy=copy_dir, to_symlink=symlink_dir
    )
    preview_manifest = Manifest(ensemble)

    # Call preview renderer for testing output
    output = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Tagged Files for model configuration" in output
    assert "Copy files" in output
    assert "Symlink" in output
    assert "Ensemble parameters" in output
    assert "Model parameters" in output

    assert "generator_files/test_dir" in output
    assert "generator_files/to_copy_dir" in output
    assert "generator_files/to_symlink_dir" in output
    assert "/SmartSim" in output
    for model in ensemble:
        assert "generator_files/test_dir" in model.files.tagged[0]
        for copy in model.files.copy:
            assert "generator_files/to_copy_dir" in copy
        for link in model.files.link:
            assert "generator_files/to_symlink_dir" in link


def test_preview_colocated_db_model_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """Test preview of DBModel on colocated ensembles, first adding the DBModel to the
    ensemble, then colocating DB.
    """

    exp_name = "test-colocated-db-model-ensemble-reordered"
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = 1

    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create the ensemble of two identical SmartSim Model
    colo_ensemble = exp.create_ensemble(
        "colocated_ens", run_settings=colo_settings, replicas=2
    )

    # Create colocated SmartSim Model
    colo_model = exp.create_model("colocated_model", colo_settings)

    # Create and save ML model to filesystem
    content = "empty test"
    path = pathlib.Path(test_dir) / "model1.pt"
    with open(path, "w", encoding="utf-8") as prev_file:
        prev_file.write(content)

    model_path = test_dir + "/model1.pt"

    # Test adding a model from ensemble
    colo_ensemble.add_ml_model(
        "cnn",
        "TF",
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs="args_0",
        outputs="Identity",
    )

    # Colocate a database with the first ensemble members
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db_tcp(
            port=test_port + i, db_cpus=1, debug=True, ifname=test_interface
        )
        # Add ML models to each ensemble member to make sure they
        # do not conflict with other ML models
        entity.add_ml_model(
            "cnn2",
            "TF",
            model_path=model_path,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
            inputs="args_0",
            outputs="Identity",
        )
        entity.disable_key_prefixing()

    # Add another ensemble member
    colo_ensemble.add_model(colo_model)

    # Colocate a database with the new ensemble member
    colo_model.colocate_db_tcp(
        port=test_port + len(colo_ensemble) - 1,
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )
    # Add a ML model to the new ensemble member
    colo_model.add_ml_model(
        "cnn2",
        "TF",
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs="args_0",
        outputs="Identity",
    )

    exp.generate(colo_ensemble)

    preview_manifest = Manifest(colo_ensemble)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Models" in output
    assert "Model name" in output
    assert "Backend" in output
    assert "Model path" in output
    assert "Device" in output
    assert "Devices per node" in output
    assert "First device" in output
    assert "Inputs" in output
    assert "Outputs" in output

    assert "cnn" in output
    assert "Model path" in output
    assert "/model1.pt" in output
    assert "CPU" in output
    assert "args_0" in output
    assert "Identity" in output


def test_preview_colocated_db_script_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """
    Test preview of DB Scripts on colocated DB from ensemble
    """

    exp_name = "test-colocated-db-script"

    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create SmartSim Ensemble with two identical models
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Create a SmartSim model
    colo_model = exp.create_model("colocated_model", colo_settings)

    # Colocate a db with each ensemble entity and add a script
    # to each entity via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_db_tcp(
            port=test_port + i,
            db_cpus=1,
            debug=True,
            ifname=test_interface,
        )

        entity.add_script(
            "test_script1",
            script_path=torch_script,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )

    # Colocate a db with the non-ensemble Model
    colo_model.colocate_db_tcp(
        port=test_port + len(colo_ensemble),
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add a script to the non-ensemble model
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add the third SmartSim model to the ensemble
    colo_ensemble.add_model(colo_model)

    # Add another script via file to the entire ensemble
    colo_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    preview_manifest = Manifest(colo_ensemble)

    # Execute preview method
    output = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Scripts" in output
    assert "Script name" in output
    assert "Script path" in output
    assert "Devices per node" in output

    assert "test_script2" in output
    assert "/torchscript.py" in output
    assert "CPU " in output


def test_preview_models_and_ensembles_html_format(test_dir, wlmutils):
    """
    Test preview of ensmebels and models in html format
    """
    # Prepare entities
    exp_name = "test-model-and-ensemble"
    test_dir = pathlib.Path(test_dir) / exp_name
    test_dir.mkdir(parents=True)
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher=test_launcher)

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)
    hello_ensemble = exp.create_ensemble("echo-ensemble", run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model, hello_ensemble)

    # Call preview renderer for testing output
    output = previewrenderer.render(exp, preview_manifest, output_format="html")

    # Evaluate output
    assert "Models" in output
    assert "echo-hello" in output
    assert "echo-spam" in output

    assert "Ensembles" in output
    assert "echo-ensemble_1" in output
    assert "echo-ensemble_2" in output


def test_output_format_error():
    """
    Test error when invalid ouput format is given.
    """
    # Prepare entities
    exp_name = "test_output_format"
    exp = Experiment(exp_name)

    # Execute preview method
    with pytest.raises(PreviewFormatError) as ex:
        exp.preview(output_format="hello")
    assert (
        "The only valid output format currently available is html" in ex.value.args[0]
    )
