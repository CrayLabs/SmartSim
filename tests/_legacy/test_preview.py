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
import typing as t
from os import path as osp

import jinja2
import numpy as np
import pytest

import smartsim
import smartsim._core._cli.utils as _utils
from smartsim import Experiment
from smartsim._core import Manifest, preview_renderer
from smartsim._core.config import CONFIG
from smartsim._core.control.controller import Controller
from smartsim._core.control.job import Job
from smartsim.database import FeatureStore
from smartsim.entity.entity import SmartSimEntity
from smartsim.error.errors import PreviewFormatError
from smartsim.settings import QsubBatchSettings, RunSettings

pytestmark = pytest.mark.group_b


@pytest.fixture
def choose_host():
    def _choose_host(wlmutils, index: int = 0):
        hosts = wlmutils.get_test_hostlist()
        if hosts:
            return hosts[index]
        return None

    return _choose_host


@pytest.fixture
def preview_object(test_dir) -> t.Dict[str, Job]:
    """
    Bare bones orch
    """
    rs = RunSettings(exe="echo", exe_args="ifname=lo")
    s = SmartSimEntity(name="faux-name", path=test_dir, run_settings=rs)
    o = FeatureStore()
    o.entity = s
    s.fs_identifier = "test_fs_id"
    s.ports = [1235]
    s.num_shards = 1
    job = Job("faux-name", "faux-step-id", s, "slurm", True)
    active_fsjobs: t.Dict[str, Job] = {"mock_job": job}
    return active_fsjobs


@pytest.fixture
def preview_object_multifs(test_dir) -> t.Dict[str, Job]:
    """
    Bare bones feature store
    """
    rs = RunSettings(exe="echo", exe_args="ifname=lo")
    s = SmartSimEntity(name="faux-name", path=test_dir, run_settings=rs)
    o = FeatureStore()
    o.entity = s
    s.fs_identifier = "testfs_reg"
    s.ports = [8750]
    s.num_shards = 1
    job = Job("faux-name", "faux-step-id", s, "slurm", True)

    rs2 = RunSettings(exe="echo", exe_args="ifname=lo")
    s2 = SmartSimEntity(name="faux-name_2", path=test_dir, run_settings=rs)
    o2 = FeatureStore()
    o2.entity = s2
    s2.fs_identifier = "testfs_reg2"
    s2.ports = [8752]
    s2.num_shards = 1
    job2 = Job("faux-name_2", "faux-step-id_2", s2, "slurm", True)

    active_fsjobs: t.Dict[str, Job] = {"mock_job": job, "mock_job2": job2}
    return active_fsjobs


def add_batch_resources(wlmutils, batch_settings):
    if isinstance(batch_settings, QsubBatchSettings):
        for key, value in wlmutils.get_batch_resources().items():
            batch_settings.set_resource(key, value)


def test_get_ifname_filter():
    """Test get_ifname filter"""

    # Test input and expected output
    value_dict = (
        (["+ifname=ib0"], "ib0"),
        ("", ""),
        ("+ifnameib0", ""),
        ("=ib0", ""),
        (["_ifname=bad_if_key"], "bad_if_key"),
        (["ifname=mock_if_name"], "mock_if_name"),
        ("IFname=case_sensitive_key", ""),
        ("xfname=not_splittable", ""),
        (None, ""),
    )

    template_str = "{{ value | get_ifname }}"
    template_dict = {"ts": template_str}

    loader = jinja2.DictLoader(template_dict)
    env = jinja2.Environment(loader=loader, autoescape=True)
    env.filters["get_ifname"] = preview_renderer.get_ifname

    t = env.get_template("ts")

    for input, expected_output in value_dict:
        output = t.render(value=input)
        # assert that that filter output matches expected output
        assert output == expected_output


def test_get_fstype_filter():
    """Test get_fstype filter to extract database backend from config"""

    template_str = "{{ config | get_fstype }}"
    template_dict = {"ts": template_str}
    loader = jinja2.DictLoader(template_dict)
    env = jinja2.Environment(loader=loader, autoescape=True)
    env.filters["get_fstype"] = preview_renderer.get_fstype

    t = env.get_template("ts")
    output = t.render(config=CONFIG.database_cli)

    assert output in CONFIG.database_cli
    # Test empty input
    test_string = ""
    output = t.render(config=test_string)
    assert output == ""
    # Test empty path
    test_string = "SmartSim/smartsim/_core/bin/"
    output = t.render(config=test_string)
    assert output == ""
    # Test no hyphen
    test_string = "SmartSim/smartsim/_core/bin/rediscli"
    output = t.render(config=test_string)
    assert output == ""
    # Test no LHS
    test_string = "SmartSim/smartsim/_core/bin/redis-"
    output = t.render(config=test_string)
    assert output == ""
    # Test no RHS
    test_string = "SmartSim/smartsim/_core/bin/-cli"
    output = t.render(config=test_string)
    assert output == ""


def test_experiment_preview(test_dir, wlmutils):
    """Test correct preview output fields for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = preview_renderer.render(exp, verbosity_level="debug")

    # Evaluate output
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert set(["Experiment Name", "Experiment Path", "Launcher"]).issubset(
        summary_dict
    )


def test_experiment_preview_properties(test_dir, wlmutils):
    """Test correct preview output properties for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = preview_renderer.render(exp, verbosity_level="debug")

    # Evaluate output
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert exp.name == summary_dict["Experiment Name"]
    assert exp.exp_path == summary_dict["Experiment Path"]
    assert exp.launcher == summary_dict["Launcher"]


def test_feature_store_preview_render(test_dir, wlmutils, choose_host):
    """Test correct preview output properties for FeatureStore preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_feature_store_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # create regular database
    feature_store = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
    )
    preview_manifest = Manifest(feature_store)

    # Execute method for template rendering
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Feature Store Identifier" in output
    assert "Shards" in output
    assert "TCP/IP Port(s)" in output
    assert "Network Interface" in output
    assert "Type" in output
    assert "Executable" in output

    fs_path = _utils.get_db_path()
    if fs_path:
        fs_type, _ = fs_path.name.split("-", 1)

    assert feature_store.fs_identifier in output
    assert str(feature_store.num_shards) in output
    assert feature_store._interfaces[0] in output
    assert fs_type in output
    assert CONFIG.database_exe in output
    assert feature_store.run_command in output
    assert str(feature_store.fs_nodes) in output


def test_preview_to_file(test_dir, wlmutils):
    """
    Test that if an output_filename is given, a file
    is rendered for Experiment preview"
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_preview_output_filename"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    filename = "test_preview_output_filename.txt"
    path = pathlib.Path(test_dir) / filename
    # Execute preview method
    exp.preview(
        output_format=preview_renderer.Format.PLAINTEXT,
        output_filename=str(path),
        verbosity_level="debug",
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

    hello_world_model = exp.create_application(
        "echo-hello", run_settings=rs1, params=model_params
    )

    spam_eggs_model = exp.create_application("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)

    # Execute preview method
    rendered_preview = preview_renderer.render(
        exp, preview_manifest, verbosity_level="debug"
    )

    # Evaluate output
    assert "Model Name" in rendered_preview
    assert "Executable" in rendered_preview
    assert "Executable Arguments" in rendered_preview
    assert "Model Parameters" in rendered_preview


def test_model_preview_properties(test_dir, wlmutils):
    """
    Test correct preview output properties for Model preview
    """
    # Prepare entities
    exp_name = "test_model_preview_parameters"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    hw_name = "echo-hello"
    hw_port = 6379
    hw_password = "unbreakable_password"
    hw_rs = "multi_tags_template.sh"
    model_params = {"port": hw_port, "password": hw_password}
    hw_param1 = "bash"
    rs1 = RunSettings(hw_param1, hw_rs)

    se_name = "echo-spam"
    se_param1 = "echo"
    se_param2 = "spam"
    se_param3 = "eggs"
    rs2 = exp.create_run_settings(se_param1, [se_param2, se_param3])

    hello_world_model = exp.create_application(
        hw_name, run_settings=rs1, params=model_params
    )
    spam_eggs_model = exp.create_application(se_name, run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)

    # Execute preview method
    rendered_preview = preview_renderer.render(
        exp, preview_manifest, verbosity_level="debug"
    )

    # Evaluate output for hello world model
    assert hw_name in rendered_preview
    assert hw_param1 in rendered_preview
    assert hw_rs in rendered_preview
    assert "port" in rendered_preview
    assert "password" in rendered_preview
    assert str(hw_port) in rendered_preview
    assert hw_password in rendered_preview

    assert hw_name == hello_world_model.name
    assert hw_param1 in hello_world_model.run_settings.exe[0]
    assert hw_rs == hello_world_model.run_settings.exe_args[0]
    assert None == hello_world_model.batch_settings
    assert "port" in list(hello_world_model.params.items())[0]
    assert str(hw_port) in list(hello_world_model.params.items())[0]
    assert "password" in list(hello_world_model.params.items())[1]
    assert hw_password in list(hello_world_model.params.items())[1]

    # Evaluate outputfor spam eggs model
    assert se_name in rendered_preview
    assert se_param1 in rendered_preview
    assert se_param2 in rendered_preview
    assert se_param3 in rendered_preview

    assert se_name == spam_eggs_model.name
    assert se_param1 in spam_eggs_model.run_settings.exe[0]
    assert se_param2 == spam_eggs_model.run_settings.exe_args[0]
    assert se_param3 == spam_eggs_model.run_settings.exe_args[1]


def test_preview_model_tagged_files(fileutils, test_dir, wlmutils):
    """
    Test model with tagged files in preview.
    """
    # Prepare entities
    exp_name = "test_model_preview_parameters"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    model_params = {"port": 6379, "password": "unbreakable_password"}
    model_settings = RunSettings("bash", "multi_tags_template.sh")

    hello_world_model = exp.create_application(
        "echo-hello", run_settings=model_settings, params=model_params
    )

    config = fileutils.get_test_conf_path(
        osp.join("generator_files", "multi_tags_template.sh")
    )
    hello_world_model.attach_generator_files(to_configure=[config])
    exp.generate(hello_world_model, overwrite=True)

    preview_manifest = Manifest(hello_world_model)

    # Execute preview method
    rendered_preview = preview_renderer.render(
        exp, preview_manifest, verbosity_level="debug"
    )

    # Evaluate output
    assert "Tagged Files for Model Configuration" in rendered_preview
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

    fs = exp.create_feature_store(port=6780, interface="lo")
    exp.generate(fs, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    model = exp.create_application("model_test", run_settings=rs1)

    # enable key prefixing on model
    model.enable_key_prefixing()
    exp.generate(model, overwrite=True)

    preview_manifest = Manifest(fs, model)

    # Execute preview method
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Key Prefix" in output
    assert "model_test" in output
    assert "Outgoing Key Collision Prevention (Key Prefixing)" in output
    assert "Tensors: On" in output
    assert "Datasets: On" in output
    assert "ML Models/Torch Scripts: Off" in output
    assert "Aggregation Lists: On" in output


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
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Ensemble Name" in output
    assert "Members" in output
    assert "Ensemble Parameters" in output


def test_preview_models_and_ensembles(test_dir, wlmutils):
    """
    Test preview of separate model entity and ensemble entity
    """
    exp_name = "test-preview-model-and-ensemble"
    test_dir = pathlib.Path(test_dir) / exp_name
    test_dir.mkdir(parents=True)
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher=test_launcher)

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hw_name = "echo-hello"
    se_name = "echo-spam"
    ens_name = "echo-ensemble"
    hello_world_model = exp.create_application(hw_name, run_settings=rs1)
    spam_eggs_model = exp.create_application(se_name, run_settings=rs2)
    hello_ensemble = exp.create_ensemble(ens_name, run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model, hello_ensemble)
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Models" in output
    assert hw_name in output
    assert se_name in output

    assert "Ensembles" in output
    assert ens_name + "_1" in output
    assert ens_name + "_2" in output


def test_ensemble_preview_client_configuration(test_dir, wlmutils):
    """
    Test preview of client configuration and key prefixing in Ensemble preview
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(
        "test-preview-ensemble-clientconfig", exp_path=test_dir, launcher=test_launcher
    )
    # Create Orchestrator
    fs = exp.create_feature_store(port=6780, interface="lo")
    exp.generate(fs, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    # Create ensemble
    ensemble = exp.create_ensemble("fd_simulation", run_settings=rs1, replicas=2)
    # enable key prefixing on ensemble
    ensemble.enable_key_prefixing()
    exp.generate(ensemble, overwrite=True)
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])
    # Create model
    ml_model = exp.create_application("tf_training", rs2)

    for sim in ensemble.entities:
        ml_model.register_incoming_entity(sim)

    exp.generate(ml_model, overwrite=True)
    preview_manifest = Manifest(fs, ml_model, ensemble)

    # Call preview renderer for testing output
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Client Configuration" in output
    assert "Feature Store Identifier" in output
    assert "Feature Store Backend" in output
    assert "Type" in output


def test_ensemble_preview_client_configuration_multifs(test_dir, wlmutils):
    """
    Test preview of client configuration and key prefixing in Ensemble preview
    with multiple feature stores
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(
        "test-preview-multifs-clinet-config", exp_path=test_dir, launcher=test_launcher
    )
    # Create feature store
    fs1_fsid = "fs_1"
    fs1 = exp.create_feature_store(port=6780, interface="lo", fs_identifier=fs1_fsid)
    exp.generate(fs1, overwrite=True)
    # Create another feature store
    fs2_fsid = "fs_2"
    fs2 = exp.create_feature_store(port=6784, interface="lo", fs_identifier=fs2_fsid)
    exp.generate(fs2, overwrite=True)

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    # Create ensemble
    ensemble = exp.create_ensemble("fd_simulation", run_settings=rs1, replicas=2)
    # enable key prefixing on ensemble
    ensemble.enable_key_prefixing()
    exp.generate(ensemble, overwrite=True)
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])
    # Create model
    ml_model = exp.create_application("tf_training", rs2)
    for sim in ensemble.entities:
        ml_model.register_incoming_entity(sim)
    exp.generate(ml_model, overwrite=True)
    preview_manifest = Manifest(fs1, fs2, ml_model, ensemble)

    # Call preview renderer for testing output
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Client Configuration" in output
    assert "Feature Store Identifier" in output
    assert "Feature Store Backend" in output
    assert "TCP/IP Port(s)" in output
    assert "Type" in output

    assert fs1_fsid in output
    assert fs2_fsid in output


def test_ensemble_preview_attached_files(fileutils, test_dir, wlmutils):
    """
    Test the preview of tagged, copy, and symlink files attached
    to an ensemble
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(
        "test-preview-attached-files", exp_path=test_dir, launcher=test_launcher
    )
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
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Tagged Files for Model Configuration" in output
    assert "Copy Files" in output
    assert "Symlink" in output
    assert "Ensemble Parameters" in output
    assert "Model Parameters" in output

    assert "generator_files/test_dir" in output
    assert "generator_files/to_copy_dir" in output
    assert "generator_files/to_symlink_dir" in output

    for model in ensemble:
        assert "generator_files/test_dir" in model.files.tagged[0]
        for copy in model.files.copy:
            assert "generator_files/to_copy_dir" in copy
        for link in model.files.link:
            assert "generator_files/to_symlink_dir" in link


def test_preview_colocated_fs_model_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """
    Test preview of FSModel on colocated ensembles
    """

    exp_name = "test-preview-colocated-fs-model-ensemble"
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
    colo_model = exp.create_application("colocated_model", colo_settings)

    # Create and save ML model to filesystem
    content = "empty test"
    model_path = pathlib.Path(test_dir) / "model1.pt"
    model_path.write_text(content)

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

    # Colocate a feature store with the first ensemble members
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i, fs_cpus=1, debug=True, ifname=test_interface
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

    # Colocate a feature store with the new ensemble member
    colo_model.colocate_fs_tcp(
        port=test_port + len(colo_ensemble) - 1,
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )
    # Add a ML model to the new ensemble member
    model_inputs = "args_0"
    model_outputs = "Identity"
    model_name = "cnn2"
    model_backend = "TF"
    colo_model.add_ml_model(
        model_name,
        model_backend,
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=model_inputs,
        outputs=model_outputs,
    )

    exp.generate(colo_ensemble)

    preview_manifest = Manifest(colo_ensemble)

    # Execute preview method
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Models" in output
    assert "Name" in output
    assert "Backend" in output
    assert "Path" in output
    assert "Device" in output
    assert "Devices Per Node" in output
    assert "Inputs" in output
    assert "Outputs" in output

    assert model_name in output
    assert model_backend in output
    assert "Path" in output
    assert "/model1.pt" in output
    assert "CPU" in output
    assert model_inputs in output
    assert model_outputs in output


def test_preview_colocated_fs_script_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """
    Test preview of FS Scripts on colocated FS from ensemble
    """

    exp_name = "test-preview-colocated-fs-script"

    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    expected_torch_script = "torchscript.py"
    test_script = fileutils.get_test_conf_path("run_fsscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path(expected_torch_script)

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
    colo_model = exp.create_application("colocated_model", colo_settings)

    # Colocate a fs with each ensemble entity and add a script
    # to each entity via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_fs_tcp(
            port=test_port + i,
            fs_cpus=1,
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

    # Colocate a fs with the non-ensemble Model
    colo_model.colocate_fs_tcp(
        port=test_port + len(colo_ensemble),
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add a script to the non-ensemble model
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    cm_name2 = "test_script2"
    colo_ensemble.add_script(
        cm_name2,
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add the third SmartSim model to the ensemble
    colo_ensemble.add_model(colo_model)

    # Add another script via file to the entire ensemble
    cm_name1 = "test_script1"
    colo_model.add_script(
        cm_name1,
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._fs_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._fs_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    preview_manifest = Manifest(colo_ensemble)

    # Execute preview method
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Torch Scripts" in output
    assert "Name" in output
    assert "Path" in output
    assert "Devices Per Node" in output

    assert cm_name2 in output
    assert expected_torch_script in output
    assert test_device in output
    assert cm_name1 in output


def test_preview_active_infrastructure(wlmutils, test_dir, preview_object):
    """Test active infrastructure without other feature stores"""

    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_active_infrastructure_preview"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = preview_renderer.render(
        exp, active_fsjobs=preview_object, verbosity_level="debug"
    )

    assert "Active Infrastructure" in output
    assert "Feature Store Identifier" in output
    assert "Shards" in output
    assert "Network Interface" in output
    assert "Type" in output
    assert "TCP/IP" in output


def test_preview_orch_active_infrastructure(
    wlmutils, test_dir, choose_host, preview_object
):
    """
    Test correct preview output properties for active infrastructure preview
    with other feature stores
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_feature_store_active_infrastructure_preview"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    feature_store2 = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        fs_identifier="fs_2",
    )

    feature_store3 = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        fs_identifier="fs_3",
    )

    preview_manifest = Manifest(feature_store2, feature_store3)

    # Execute method for template rendering
    output = preview_renderer.render(
        exp, preview_manifest, active_fsjobs=preview_object, verbosity_level="debug"
    )

    assert "Active Infrastructure" in output
    assert "Feature Store Identifier" in output
    assert "Shards" in output
    assert "Network Interface" in output
    assert "Type" in output
    assert "TCP/IP" in output


def test_preview_multifs_active_infrastructure(
    wlmutils, test_dir, choose_host, preview_object_multidb
):
    """multiple started feature stores active infrastructure"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # start a new Experiment for this section
    exp = Experiment(
        "test_preview_multifs_active_infrastructure",
        exp_path=test_dir,
        launcher=test_launcher,
    )

    # Execute method for template rendering
    output = preview_renderer.render(
        exp, active_fsjobs=preview_object_multifs, verbosity_level="debug"
    )

    assert "Active Infrastructure" in output
    assert "Feature Store Identifier" in output
    assert "Shards" in output
    assert "Network Interface" in output
    assert "Type" in output
    assert "TCP/IP" in output

    assert "testfs_reg" in output
    assert "testfs_reg2" in output
    assert "Feature Stores" not in output


def test_preview_active_infrastructure_feature_store_error(
    wlmutils, test_dir, choose_host, monkeypatch: pytest.MonkeyPatch
):
    """Demo error when trying to preview a started feature store"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_active_infrastructure_preview_orch_error"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    monkeypatch.setattr(
        smartsim.database.orchestrator.FeatureStore, "is_active", lambda x: True
    )

    orc = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        fs_identifier="orc_1",
    )

    # Retrieve any active jobs
    active_fsjobs = exp._control.active_feature_store_jobs

    preview_manifest = Manifest(orc)

    # Execute method for template rendering
    output = preview_renderer.render(
        exp, preview_manifest, active_fsjobs=active_fsjobs, verbosity_level="debug"
    )

    assert "WARNING: Cannot preview orc_1, because it is already started" in output


def test_active_feature_store_jobs_property(
    wlmutils,
    test_dir,
    preview_object,
):
    """Ensure fs_jobs remaines unchanged after deletion
    of active_feature_store_jobs property stays intact when retrieving fs_jobs"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment for this section
    exp = Experiment(
        "test-active_feature_store_jobs-property",
        exp_path=test_dir,
        launcher=test_launcher,
    )

    controller = Controller()
    controller._jobs.fs_jobs = preview_object

    # Modify the returned job collection
    active_feature_store_jobs = exp._control.active_feature_store_jobs
    active_feature_store_jobs["test"] = "test_value"

    # Verify original collection is not also modified
    assert not exp._control.active_feature_store_jobs.get("test", None)


def test_verbosity_info_ensemble(test_dir, wlmutils):
    """
    Test preview of separate model entity and ensemble entity
    with verbosity level set to info
    """
    exp_name = "test-model-and-ensemble"
    test_dir = pathlib.Path(test_dir) / exp_name
    test_dir.mkdir(parents=True)
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher=test_launcher)

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hw_name = "echo-hello"
    se_name = "echo-spam"
    ens_name = "echo-ensemble"
    hello_world_model = exp.create_application(hw_name, run_settings=rs1)
    spam_eggs_model = exp.create_application(se_name, run_settings=rs2)
    hello_ensemble = exp.create_ensemble(ens_name, run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model, hello_ensemble)
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="info")

    assert "Executable" not in output
    assert "Executable Arguments" not in output

    assert "echo_ensemble_1" not in output


def test_verbosity_info_colocated_fs_model_ensemble(
    fileutils, test_dir, wlmutils, mlutils
):
    """Test preview of FSModel on colocated ensembles, first adding the FSModel to the
    ensemble, then colocating FS.
    """

    exp_name = "test-colocated-fs-model-ensemble-reordered"
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
    colo_model = exp.create_application("colocated_model", colo_settings)

    # Create and save ML model to filesystem
    content = "empty test"
    model_path = pathlib.Path(test_dir) / "model1.pt"
    model_path.write_text(content)

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

    # Colocate a feature store with the first ensemble members
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i, fs_cpus=1, debug=True, ifname=test_interface
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

    # Colocate a feature store with the new ensemble member
    colo_model.colocate_fs_tcp(
        port=test_port + len(colo_ensemble) - 1,
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )
    # Add a ML model to the new ensemble member
    model_inputs = "args_0"
    model_outputs = "Identity"
    model_name = "cnn2"
    model_backend = "TF"
    colo_model.add_ml_model(
        model_name,
        model_backend,
        model_path=model_path,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
        inputs=model_inputs,
        outputs=model_outputs,
    )

    exp.generate(colo_ensemble)

    preview_manifest = Manifest(colo_ensemble)

    # Execute preview method
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="info")

    assert "Outgoing Key Collision Prevention (Key Prefixing)" not in output
    assert "Devices Per Node" not in output


def test_verbosity_info_feature_store(test_dir, wlmutils, choose_host):
    """Test correct preview output properties for feature store preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_feature_store_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # create regular feature store
    feature_store = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
    )
    preview_manifest = Manifest(feature_store)

    # Execute method for template rendering
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="info")

    # Evaluate output
    assert "Executable" not in output
    assert "Run Command" not in output


def test_verbosity_info_ensemble(test_dir, wlmutils):
    """
    Test client configuration and key prefixing in Ensemble preview
    """
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment("key_prefix_test", exp_path=test_dir, launcher=test_launcher)
    # Create feature store
    fs = exp.create_feature_store(port=6780, interface="lo")
    exp.generate(fs, overwrite=True)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    # Create ensemble
    ensemble = exp.create_ensemble("fd_simulation", run_settings=rs1, replicas=2)
    # enable key prefixing on ensemble
    ensemble.enable_key_prefixing()
    exp.generate(ensemble, overwrite=True)
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])
    # Create model
    ml_model = exp.create_application("tf_training", rs2)

    for sim in ensemble.entities:
        ml_model.register_incoming_entity(sim)

    exp.generate(ml_model, overwrite=True)
    preview_manifest = Manifest(fs, ml_model, ensemble)

    # Call preview renderer for testing output
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="info")

    # Evaluate output
    assert "Outgoing Key Collision Prevention (Key Prefixing)" in output


def test_check_output_format_error():
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
        "The only valid output format currently available is plain_text"
        in ex.value.args[0]
    )


def test_check_verbosity_level_error():
    """
    Testing that an error does occur when a string verbosity is passed
    """
    # Prepare entities
    exp_name = "test_verbosity_level_error"
    exp = Experiment(exp_name)

    # Execute preview method
    with pytest.raises(ValueError) as ex:
        exp.preview(verbosity_level="hello")


def test_check_verbosity_level():
    """
    Testing that an error doesnt occur when a string verbosity is passed
    """
    # Prepare entities
    exp_name = "test_verbosity_level"
    exp = Experiment(exp_name)

    # Execute preview method
    exp.preview(verbosity_level="info")


def test_preview_colocated_fs_singular_model(wlmutils, test_dir):
    """Test preview behavior when a colocated fs is only added to
    one model. The expected behviour is that both models are colocated
    """

    test_launcher = wlmutils.get_test_launcher()

    exp = Experiment("colocated test", exp_path=test_dir, launcher=test_launcher)

    rs = exp.create_run_settings("sleep", ["100"])

    model_1 = exp.create_application("model_1", run_settings=rs)
    model_2 = exp.create_application("model_2", run_settings=rs)

    model_1.colocate_fs()

    exp.generate(model_1, model_2, overwrite=True)

    preview_manifest = Manifest(model_1, model_2)

    # Call preview renderer for testing output
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    assert "model_1" in output
    assert "model_2" in output
    assert "Client Configuration" in output


def test_preview_fs_script(wlmutils, test_dir):
    """
    Test preview of model instance with a torch script.
    """
    test_launcher = wlmutils.get_test_launcher()
    # Initialize the Experiment and set the launcher to auto

    exp = Experiment("getting-started", launcher=test_launcher)

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe="python", exe_args="params.py")

    # Initialize a Model object
    model_instance = exp.create_application("model_name", model_settings)
    model_instance.colocate_fs_tcp()

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
    preview_manifest = Manifest(model_instance)

    # Call preview renderer for testing output
    output = preview_renderer.render(exp, preview_manifest, verbosity_level="debug")

    # Evaluate output
    assert "Torch Script" in output
