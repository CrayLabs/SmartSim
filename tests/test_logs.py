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
import conftest
import logging
import os
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
    logger.warn("warn")
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
    logger.warn("warn")
    logger.error("error")
    logger.exception("exception")

    logged_messages = faux_out_stream.getvalue().split("\n")
    assert 'debug' in logged_messages
    assert 'info' in logged_messages
    assert 'warn' in logged_messages
    assert 'error' not in logged_messages
    assert 'exception' not in logged_messages


def test_add_exp_loggers(fileutils: conftest.FileUtils):
    """Ensure that expected loggers are added"""
    test_dir = fileutils.make_test_dir()
    faux_out_stream = io.StringIO()

    logger = logging.getLogger("smartsim_test_add_exp_loggers")
    logger.addHandler(logging.StreamHandler(faux_out_stream))
    
    smartsim.log.add_exp_loggers(test_dir, logger)

    logger.debug("debug")
    logger.exception("exception")

    out_path = os.path.join(test_dir, "smartsim.out")
    err_path = os.path.join(test_dir, "smartsim.err")

    assert os.path.exists(out_path)
    assert os.path.exists(err_path)
