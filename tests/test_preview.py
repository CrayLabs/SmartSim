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

from smartsim import Experiment


def test_experiment_preview(test_dir, wlmutils):
    """Test correct preview output items for Experiment preview"""
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # Call method for string formatting for testing
    output = exp._preview().strip()
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "") for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert set(["Experiment", "Experiment Path", "Launcher"]).issubset(summary_dict)


def test_experiment_preview_properties(test_dir, wlmutils):
    """Test correct preview output properties for Experiment preview"""
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # Call method for string formatting for testing
    output = exp._preview().strip()
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "") for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert exp.name == summary_dict["Experiment"]
    assert exp.exp_path == summary_dict["Experiment Path"]
    assert exp.get_launcher() == summary_dict["Launcher"]

def test_output_format_error():
    exp_name = "test_output_format"
    exp = Experiment(exp_name)

    with pytest.raises(NotImplementedError):
        exp.preview(output_format="hello")


def test_output_filename_error():
    exp_name = "test_output_filename"
    exp = Experiment(exp_name)

    with pytest.raises(NotImplementedError):
        exp.preview(output_filename="hello")


def test_verbosity_level_error():
    exp_name = "test_verbosity_level"
    exp = Experiment(exp_name)

    with pytest.raises(NotImplementedError):
        exp.preview(verbosity_level="hello")
