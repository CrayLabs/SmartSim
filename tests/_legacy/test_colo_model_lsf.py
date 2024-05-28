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

import sys

import pytest

import smartsim.settings.base
from smartsim import Experiment
from smartsim.entity import Model
from smartsim.settings.lsfSettings import JsrunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


is_mac = sys.platform == "darwin"


class ExpectationMet(Exception):
    """Use this to break a test when we verify a call path is achieved"""

    ...


def show_expectation_met(*args, **kwargs):
    raise ExpectationMet("mock._prep_colocated_db")


def test_jsrun_prep(fileutils, coloutils, monkeypatch):
    """Ensure that JsrunSettings prep method is executed as expected"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")
    # mock the prep method to raise an exception that short circuits test when goal is met
    monkeypatch.setattr(JsrunSettings, "_prep_colocated_db", show_expectation_met)

    db_args = {"custom_pinning": [1]}
    db_type = "uds"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    with pytest.raises(ExpectationMet, match="mock._prep_colocated_db") as ex:
        run_settings = JsrunSettings("foo")

        coloutils.setup_test_colo(
            fileutils,
            db_type,
            exp,
            "send_data_local_smartredis.py",
            db_args,
            colo_settings=run_settings,
        )


def test_non_js_run_prep(fileutils, coloutils, monkeypatch):
    """Ensure that RunSettings does not attempt to call a prep method"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")
    # mock prep method to ensure that the exception isn't thrown w/non-JsrunSettings arg
    monkeypatch.setattr(JsrunSettings, "_prep_colocated_db", show_expectation_met)

    db_args = {"custom_pinning": [1]}
    db_type = "tcp"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    run_settings = smartsim.settings.base.RunSettings("foo")

    colo_model: Model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
        colo_settings=run_settings,
    )

    assert colo_model


@pytest.mark.parametrize(
    "exp_run_arg_key,run_arg_key,exp_value,test_value",
    [
        pytest.param("cpu_per_rs", "cpu_per_rs", 11, 11, id="cpu_per_rs matches input"),
        pytest.param("c", "c", 22, 22, id="c matches input"),
        pytest.param(
            "cpu_per_rs", "cpu_per_rsx", 1, 33, id="key typo: cpu_per_rsx gives default"
        ),
        pytest.param("cpu_per_rs", "cx", 1, 44, id="key typo: cx gives default"),
    ],
)
def test_jsrun_prep_cpu_per_flag_set_check(
    fileutils,
    coloutils,
    monkeypatch,
    exp_run_arg_key,
    run_arg_key,
    exp_value,
    test_value,
):
    """Ensure that _prep_colocated_db honors basic cpu_per_rs config and allows a
    valid input parameter to result in the correct output. If no expected input (or
    incorrect key) is given, the default should be returned using default config key"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")

    # excluding "db_cpus" should result in default value in comparison & output
    db_args = {"custom_pinning": [1]}
    db_type = "uds"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    run_args = {run_arg_key: test_value}
    run_settings = JsrunSettings("foo", run_args=run_args)

    colo_model: Model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
        colo_settings=run_settings,
    )

    assert colo_model.run_settings.run_args[exp_run_arg_key] == exp_value


@pytest.mark.parametrize(
    "exp_run_arg_key,run_arg_key,exp_value,test_value",
    [
        pytest.param("cpu_per_rs", "cpu_per_rs", 11, 11, id="cpu_per_rs matches input"),
        pytest.param("c", "c", 22, 22, id="c matches input"),
        pytest.param(
            "cpu_per_rs", "cpu_per_rsx", 3, 33, id="key typo: db_cpus out (not default)"
        ),
        pytest.param(
            "cpu_per_rs", "cx", 3, 44, id="key typo: get db_cpus out (not default)"
        ),
    ],
)
def test_jsrun_prep_db_cpu_override(
    fileutils,
    coloutils,
    monkeypatch,
    exp_run_arg_key,
    run_arg_key,
    exp_value,
    test_value,
):
    """Ensure that both cpu_per_rs and c input config override db_cpus"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")

    # setting "db_cpus" should result in non-default value in comparison & output
    db_args = {"custom_pinning": [1], "db_cpus": 3}
    db_type = "tcp"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    run_args = {run_arg_key: test_value}
    run_settings = JsrunSettings("foo", run_args=run_args)

    colo_model: Model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
        colo_settings=run_settings,
    )

    assert colo_model.run_settings.run_args[exp_run_arg_key] == exp_value


@pytest.mark.parametrize(
    "exp_run_arg_key,run_arg_key,exp_value,test_value",
    [
        pytest.param(
            "cpu_per_rs", "cpu_per_rs", 8, 3, id="cpu_per_rs swaps to db_cpus"
        ),
        pytest.param("c", "c", 8, 4, id="c swaps to db_cpus"),
        pytest.param("cpu_per_rs", "cpu_per_rsx", 8, 5, id="key typo: db_cpus out"),
        pytest.param("cpu_per_rs", "cx", 8, 6, id="key typo: get db_cpus out"),
    ],
)
def test_jsrun_prep_db_cpu_replacement(
    fileutils,
    coloutils,
    monkeypatch,
    exp_run_arg_key,
    run_arg_key,
    exp_value,
    test_value,
):
    """Ensure that db_cpus default is used if user config suggests underutilizing resources"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")

    # setting "db_cpus" should result in non-default value in comparison & output
    db_args = {"custom_pinning": [1], "db_cpus": 8}
    db_type = "uds"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    run_args = {run_arg_key: test_value}
    run_settings = JsrunSettings("foo", run_args=run_args)

    colo_model: Model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
        colo_settings=run_settings,
    )

    assert colo_model.run_settings.run_args[exp_run_arg_key] == exp_value


@pytest.mark.parametrize(
    "exp_run_arg_key,run_arg_key,exp_value,test_value",
    [
        pytest.param("rs_per_host", "rs_per_host", 1, 1, id="rs_per_host is 1"),
        pytest.param("r", "r", 1, 1, id="r is 1"),
        pytest.param("rs_per_host", "rs_per_host", 1, 2, id="rs_per_host replaced w/1"),
        pytest.param("r", "r", 1, 3, id="r replaced w/1"),
        pytest.param(
            "rs_per_host",
            "rs_per_hostx",
            1,
            4,
            id="key typo: rs_per_hostx gets default",
        ),
        pytest.param("rs_per_host", "rx", 1, 5, id="key typo: rx gets default"),
    ],
)
def test_jsrun_prep_rs_per_host(
    fileutils,
    coloutils,
    monkeypatch,
    exp_run_arg_key,
    run_arg_key,
    exp_value,
    test_value,
):
    """Ensure that resource-per-host settings are configured and are modified as
    required to meet limitations (e.g. rs_per_host MUST equal 1)"""
    monkeypatch.setattr(smartsim.settings.base, "expand_exe_path", lambda x: "/bin/{x}")

    db_args = {"custom_pinning": [1]}
    db_type = "tcp"  # Test is insensitive to choice of db

    exp = Experiment("colocated_model_lsf", launcher="lsf")

    run_args = {run_arg_key: test_value}
    run_settings = JsrunSettings("foo", run_args=run_args)

    colo_model: Model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
        colo_settings=run_settings,
    )

    # NOTE: _prep_colocated_db sets this to a string & not an integer
    assert str(colo_model.run_settings.run_args[exp_run_arg_key]) == str(exp_value)
