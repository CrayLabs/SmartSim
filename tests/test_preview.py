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
from os import path as osp

import pytest

import smartsim._core._cli.utils as _utils
from smartsim import Experiment
from smartsim._core import Manifest, previewrenderer
from smartsim._core.config import CONFIG
from smartsim.error.errors import PreviewFormatError


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


def test_orchestrator_preview_render(test_dir, wlmutils, choose_host):
    """Test correct preview output properties for Orchestrator preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_experiment_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # create regular database
    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
    )
    preview_manifest = Manifest(orc)

    # Execute method for template rendering
    output = previewrenderer.render(exp, preview_manifest)

    # Evaluate output
    assert "Database identifier" in output
    assert "Shards" in output
    assert "TCP/IP port" in output
    assert "Network interface" in output
    assert "Type" in output
    assert "Executable" in output
    assert "Batch Launch" in output
    assert "Run command" in output
    assert "Ntasks" in output

    db_path = _utils.get_db_path()
    if db_path:
        db_type, _ = db_path.name.split("-", 1)

    assert orc.db_identifier in output
    assert str(orc.num_shards) in output
    assert orc._interfaces[0] in output
    assert db_type in output
    assert CONFIG.database_exe in output
    assert str(orc.batch) in output
    assert orc.run_command in output
    assert str(orc.db_nodes) in output


def test_preview_to_file(test_dir, wlmutils, fileutils):
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
    exp.preview(output_format="plain_text", output_filename=str(path))

    # Evaluate output
    assert path.exists()
    assert path.is_file()


def test_orchestrator_preview_output_format_html(test_dir, wlmutils, choose_host):
    """Test that an html file is rendered for Orchestrator preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_orchestrator_preview_output_format_html"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
    )
    filename = "test_orchestrator_preview_output_format_html.html"
    path = pathlib.Path(test_dir) / filename

    # Execute preview method
    exp.preview(orc, output_format="html", output_filename=str(path))

    # Evaluate output
    assert path.exists()
    assert path.is_file()


def test_preview_active_infrastructure(wlmutils, test_dir, choose_host):
    """Test correct preview output properties for active infrastructure preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_active_infrastructure_preview"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        db_identifier="orc_1",
    )

    exp.start(orc)

    assert orc.is_active() == True

    orc2 = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        db_identifier="orc_2",
    )

    orc3 = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        db_identifier="orc_3",
    )

    preview_manifest = Manifest(orc, orc2, orc3)

    # Execute method for template rendering
    output = previewrenderer.render(exp, preview_manifest)
    print(output)

    assert "Active Infrastructure" in output
    assert "Database identifier" in output
    assert "Shards" in output
    assert "Network interface" in output
    assert "Type" in output
    assert "TCP/IP" in output
    assert "Orchestrators" in output

    exp.stop(orc)


def test_active_infrastructure_preview_output_format_html(
    test_dir, wlmutils, choose_host
):
    """Test that an html file is rendered for active infrastructure preview"""
    # Prepare entities
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    exp_name = "test_orchestrator_preview_output_format_html"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        db_identifier="orc_1",
    )
    exp.start(orc)
    assert orc.is_active() == True

    orc2 = exp.create_database(
        port=test_port,
        interface=test_interface,
        hosts=choose_host(wlmutils),
        db_identifier="orc_2",
    )

    assert orc2.is_active() == False

    filename = "test_active_infrastructure_preview_output_format_html.html"
    path = pathlib.Path(test_dir) / filename

    # Execute preview method
    exp.preview(orc, orc2, output_format="html", output_filename=str(path))

    # Evaluate output
    assert path.exists()
    assert path.is_file()

    exp.stop(orc)


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
        "The only valid output format currently available is plain_text"
        in ex.value.args[0]
    )