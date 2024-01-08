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

import io
import logging
import os
import pathlib
import pytest
import smartsim.log


# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


"""
Test logging features
"""


def test_level_filter_info():
    """Ensure that messages above maximum are not logged"""
    log_filter = smartsim.log.LowPassFilter("INFO")
    faux_out_stream = io.StringIO()

    logger = logging.getLogger("test_level_filter_info")
    logger.addHandler(logging.StreamHandler(faux_out_stream))
    logger.addFilter(log_filter)
    
    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("error")
    logger.exception("exception")

    logged_messages = faux_out_stream.getvalue().split("\n")
    assert 'debug' in logged_messages
    assert 'info' in logged_messages
    assert 'warn' not in logged_messages
    assert 'error' not in logged_messages
    assert 'exception' not in logged_messages


def test_level_filter_warn():
    """Ensure that messages above maximum are not logged"""
    log_filter = smartsim.log.LowPassFilter("WARN")

    faux_out_stream = io.StringIO()

    logger = logging.getLogger("test_level_filter_warn")
    logger.addHandler(logging.StreamHandler(faux_out_stream))
    logger.addFilter(log_filter)
    
    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("error")
    logger.exception("exception")

    logged_messages = faux_out_stream.getvalue().split("\n")
    assert 'debug' in logged_messages
    assert 'info' in logged_messages
    assert 'warn' in logged_messages
    assert 'error' not in logged_messages
    assert 'exception' not in logged_messages


def test_add_exp_loggers(test_dir):
    """Ensure that expected loggers are added"""
    # test_dir = fileutils.make_test_dir()
    faux_out_stream = io.StringIO()

    logger = logging.getLogger("smartsim_test_add_exp_loggers")
    logger.addHandler(logging.StreamHandler(faux_out_stream))

    filename1 = pathlib.Path(test_dir) / "smartsim.out"
    filename2 = pathlib.Path(test_dir) / "smartsim.err"

    filter_fn = lambda x: True
    
    smartsim.log.log_to_file(filename1, logger=logger, log_filter=filter_fn)
    smartsim.log.log_to_file(filename2, "WARN", logger)

    logger.debug("debug")
    logger.exception("exception")

    assert filename1.exists()
    assert filename1.is_file()

    assert filename2.exists()
    assert filename2.is_file()


def test_get_logger(test_dir: str):
    """Ensure the correct logger type is instantiated"""
    logger = smartsim.log.get_logger("SmartSimTest", "INFO")
    assert isinstance(logger, smartsim.log.ContextAwareLogger)


def test_exp_logs(test_dir: str):
    """Ensure that experiment loggers are added when context info exists"""
    test_dir = pathlib.Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    token = smartsim.log.ctx_exp_path.set(test_dir)
    
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

    out_file = pathlib.Path(test_dir) / "smartsim.out"
    err_file = pathlib.Path(test_dir) / "smartsim.err"

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

    smartsim.log.ctx_exp_path.reset(token)
