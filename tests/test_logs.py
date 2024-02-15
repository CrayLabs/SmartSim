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

import io
import logging
import pathlib

import pytest

import smartsim
import smartsim.log
from smartsim import Experiment

_CFG_TM_ENABLED_ATTR = "telemetry_enabled"

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


@pytest.fixture
def turn_on_tm(monkeypatch):
    monkeypatch.setattr(
        smartsim._core.config.config.Config,
        _CFG_TM_ENABLED_ATTR,
        property(lambda self: True),
    )
    yield


@pytest.mark.parametrize(
    "level,expect_d,expect_i,expect_w,expect_e",
    [
        pytest.param("DEBUG", True, False, False, False, id="debug-level"),
        pytest.param("INFO", True, True, False, False, id="info-level"),
        pytest.param("WARNING", True, True, True, False, id="warn-level"),
        pytest.param("ERROR", True, True, True, True, id="err-level"),
    ],
)
def test_lowpass_filter(level, expect_d, expect_i, expect_w, expect_e):
    """Ensure that messages above maximum are not logged"""
    log_filter = smartsim.log.LowPassFilter(level)

    faux_out_stream = io.StringIO()
    handler = logging.StreamHandler(faux_out_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(f"test_level_filter_{level}")
    logger.addHandler(handler)
    logger.addFilter(log_filter)

    logger.debug(str(logging.DEBUG))
    logger.info(str(logging.INFO))
    logger.warning(str(logging.WARNING))
    logger.exception(str(logging.ERROR))

    logged_messages = faux_out_stream.getvalue().split("\n")
    assert (str(logging.DEBUG) in logged_messages) == expect_d
    assert (str(logging.INFO) in logged_messages) == expect_i
    assert (str(logging.WARN) in logged_messages) == expect_w
    assert (str(logging.ERROR) in logged_messages) == expect_e


def test_add_exp_loggers(test_dir):
    """Ensure that expected loggers are added"""
    # test_dir = fileutils.make_test_dir()
    faux_out_stream = io.StringIO()

    logger = logging.getLogger("smartsim_test_add_exp_loggers")
    logger.addHandler(logging.StreamHandler(faux_out_stream))

    out_file = pathlib.Path(test_dir) / "smartsim.out"
    err_file = pathlib.Path(test_dir) / "smartsim.err"

    filter_fn = lambda x: True

    smartsim.log.log_to_exp_file(str(out_file), logger, log_filter=filter_fn)
    smartsim.log.log_to_exp_file(str(err_file), logger, "WARN")

    logger.debug("debug")
    logger.exception("exception")

    assert out_file.exists()
    assert out_file.is_file()

    assert err_file.exists()
    assert err_file.is_file()


def test_get_logger(test_dir: str, turn_on_tm, monkeypatch):
    """Ensure the correct logger type is instantiated"""
    monkeypatch.setenv("SMARTSIM_LOG_LEVEL", "developer")
    logger = smartsim.log.get_logger("SmartSimTest", "INFO")
    assert isinstance(logger, smartsim.log.ContextAwareLogger)


@pytest.mark.parametrize(
    "input_level,exp_level",
    [
        pytest.param("INFO", "info", id="lowercasing only, INFO"),
        pytest.param("info", "info", id="input back, info"),
        pytest.param("WARNING", "warning", id="lowercasing only, WARNING"),
        pytest.param("warning", "warning", id="input back, warning"),
        pytest.param("QUIET", "warning", id="lowercasing only, QUIET"),
        pytest.param("quiet", "warning", id="translation back, quiet"),
        pytest.param("DEVELOPER", "debug", id="lowercasing only, DEVELOPER"),
        pytest.param("developer", "debug", id="translation back, developer"),
    ],
)
def test_translate_log_level(input_level: str, exp_level: str, turn_on_tm):
    """Ensure the correct logger type is instantiated"""
    translated_level = smartsim.log._translate_log_level(input_level)
    assert exp_level == translated_level


def test_exp_logs(test_dir: str, turn_on_tm, monkeypatch):
    """Ensure that experiment loggers are added when context info exists"""
    monkeypatch.setenv("SMARTSIM_LOG_LEVEL", "developer")
    test_dir = pathlib.Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    token = smartsim.log.ctx_exp_path.set(test_dir)

    try:
        logger = smartsim.log.get_logger("SmartSimTest", "INFO")

        faux_out_stream = io.StringIO()
        logger.addHandler(logging.StreamHandler(faux_out_stream))

        log_msg = "testing in a test!"
        err_msg = "erroring in a test!"
        logger.info(log_msg)
        logger.error(err_msg)

        # ensure that the default stream is written to
        logged = faux_out_stream.getvalue()

        assert log_msg in logged
        assert err_msg in logged

        out_file, err_file = smartsim.log.get_exp_log_paths()

        out_content = out_file.read_text()
        err_content = err_file.read_text()

        # ensure the low-pass filter logs non-errors to out file
        assert log_msg in out_content
        assert err_msg not in out_content
        assert str(test_dir) in out_content

        # ensure the errors are logged to err file
        assert err_msg in err_content
        assert log_msg not in err_content
        assert str(err_msg) in err_content
    finally:
        smartsim.log.ctx_exp_path.reset(token)


def test_context_leak(test_dir: str, turn_on_tm, monkeypatch):
    """Ensure that exceptions do not leave the context in an invalid state"""
    test_dir = pathlib.Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    original_ctx_value = test_dir / pathlib.Path("some value")
    ctx_var = smartsim.log.ctx_exp_path
    token = ctx_var.set(original_ctx_value)

    err_msg = "some ex occurred in JobManager"

    def thrower(_self):
        raise Exception(err_msg)

    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(smartsim._core.control.jobmanager.JobManager, "start", thrower)
            exp = Experiment("MyExperiment", launcher="local", exp_path=str(test_dir))

            sleep_rs = exp.create_run_settings("sleep", ["2"])
            sleep_rs.set_nodes(1)
            sleep_rs.set_tasks(1)

            sleep = exp.create_model("SleepModel", sleep_rs)
            exp.generate(sleep)
            exp.start(sleep, block=True)
    except Exception as ex:
        assert err_msg in ex.args
    finally:
        assert ctx_var.get() == original_ctx_value
        ctx_var.reset(token)
        assert ctx_var.get() == ""
