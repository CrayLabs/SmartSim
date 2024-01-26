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
import pytest

from smartsim._core import Manifest
from smartsim import Experiment
from smartsim._core import previewrenderer
import smartsim._core._cli.utils as _utils
from smartsim._core.config import CONFIG
import pathlib
import numpy as np


@pytest.fixture
def choose_host():
    def _choose_host(wlmutils, index=0):
        hosts = wlmutils.get_test_hostlist()
        if hosts:
            return hosts[index]
        else:
            return None

    return _choose_host


def test_experiment_preview(test_dir, wlmutils):
    """Test correct preview output items for Experiment preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Execute method for template rendering
    output = previewrenderer.render(exp)
    print(output)
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


def test_ensembles_jp(test_dir, wlmutils):
    exp_name = "test-exp"
    test_dir = pathlib.Path(test_dir) / exp_name
    test_dir.mkdir(parents=True)
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher="local")

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)
    hello_ensemble = exp.create_ensemble("echo-ensemble", run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model, hello_ensemble)
    output = previewrenderer.render(exp, preview_manifest)

    # print(spam_eggs_model.run_settings.exe[0])
    # print(spam_eggs_model.run_settings.exe_args)
    # for arg in model.run_settings.exe_arg


def test_ensembles_params(test_dir, wlmutils):
    exp = Experiment("Training-Run", launcher="auto")

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
    # exp.start(ensemble, summary=True)

    # exp.generate(ensemble)

    # print(ensemble.params.keys())

    for key, value in ensemble.params.items():
        print(key, value)

    preview_manifest = Manifest(ensemble)
    output = previewrenderer.render(exp, preview_manifest)

    # logger.info(output)

    # exp.summary()


# def test_orchestrator_preview_render(test_dir, wlmutils):
#     """Test correct preview output properties for Orchestrator preview"""
#     test_launcher = wlmutils.get_test_launcher()
#     test_interface = wlmutils.get_test_interface()
#     test_port = wlmutils.get_test_port()

#     exp_name = "test_experiment_preview_properties"
#     exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

#     # create regular database
#     orc = exp.create_database(
#         port=test_port,
#         interface=test_interface,
#         hosts=choose_host(wlmutils),
#     )

#     preview_manifest = Manifest(orc)
#     output = previewrenderer.render(exp,preview_manifest)

#     assert "Database identifier" in output
#     assert "Shards" in output
#     assert "Network interface" in output
#     assert "Type" in output
#     assert "Executable" in output
#     assert "Batch Launch" in output
#     assert "Run command" in output
#     assert "Ntasks" in output

#     db_path = _utils.get_db_path()
#     if db_path:
#         db_type, _ = db_path.name.split("-", 1)

#     assert orc.db_identifier in output
#     assert str(orc.num_shards) in output
#     assert orc._interfaces[0] in output
#     assert db_type in output
#     assert CONFIG.database_exe in output
#     assert str(orc.batch) in output
#     assert orc.run_command in output
#     assert str(orc.db_nodes) in output


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


def test_model_jp(test_dir, wlmutils):
    exp_name = "test_model_jp"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)


def test_model_preview(test_dir, wlmutils):
    exp_name = "test_model_preview"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)
    assert "Model name" in rendered_preview
    assert "Executable" in rendered_preview
    assert "Executable Arguments" in rendered_preview


def test_model_preview_parameters(test_dir, wlmutils):
    exp_name = "test_model_preview_parameters"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)

    # for hello world model
    assert "echo-hello" in rendered_preview
    assert "/usr/bin/echo" in rendered_preview
    assert "hello" in rendered_preview
    assert "world" in rendered_preview
    assert "echo-hello" == hello_world_model.name
    assert "/usr/bin/echo" == hello_world_model.run_settings.exe[0]
    assert "hello" == hello_world_model.run_settings.exe_args[0]
    assert "world" == hello_world_model.run_settings.exe_args[1]
    # for spam eggs model
    assert "echo-spam" in rendered_preview
    assert "/usr/bin/echo" in rendered_preview
    assert "spam" in rendered_preview
    assert "eggs" in rendered_preview
    assert "echo-spam" == spam_eggs_model.name
    assert "/usr/bin/echo" == spam_eggs_model.run_settings.exe[0]
    assert "spam" == spam_eggs_model.run_settings.exe_args[0]
    assert "eggs" == spam_eggs_model.run_settings.exe_args[1]

    # def test_ensemble_preview_render(test_dir, wlmutils):


#     test_launcher = wlmutils.get_test_launcher()
#     test_interface = wlmutils.get_test_interface()
#     test_port = wlmutils.get_test_port()

#     exp_name = "test_experiment_preview_properties"
#     exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

#     ens_settings = exp.create_run_settings(exe="sleep", exe_args="3")
#     ensemble = exp.create_ensemble(
#         "ensemble-replica", replicas=4, run_settings=ens_settings
#     )

#     params = {"tutorial_name": ["Ellie", "John"], "tutorial_parameter": [2, 11]}
#     ensemble = exp.create_ensemble(
#         "ensemble", params=params, run_settings=rs, perm_strategy="all_perm"
#     )

#     # to_configure specifies that the files attached should be read and tags should be looked for
#     config_file = "./output_my_parameter.py"
#     ensemble.attach_generator_files(to_configure=config_file)

#     exp.generate(ensemble, overwrite=True)
#     exp.start(ensemble)


def test_output_format_error():
    """
    Test error when invalid ouput format is given.
    """
    # Prepare entities
    exp_name = "test_output_format"
    exp = Experiment(exp_name)

    # Execute preview method
    with pytest.raises(ValueError) as ex:
        exp.preview(output_format="hello")
    assert "The only valid currently available is html" in ex.value.args[0]
