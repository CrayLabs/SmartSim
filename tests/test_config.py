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


import os
from pathlib import Path

import pytest

from smartsim._core.config.config import Config
from smartsim.error import SSConfigError


def test_all_config_defaults():
    config = Config()
    assert Path(config.redisai).is_file()
    assert Path(config.database_exe).is_file()
    assert Path(config.database_cli).is_file()
    assert Path(config.database_conf).is_file()

    # these will be changed so we will just run them
    assert config.log_level
    assert config.jm_interval

    config.test_interface
    config.test_launcher
    config.test_account
    config.test_device


def test_redisai():
    config = Config()
    assert Path(config.redisai).is_file()
    assert isinstance(config.redisai, str)

    os.environ["RAI_PATH"] = "not/a/path"
    config = Config()
    with pytest.raises(SSConfigError):
        config.redisai
    os.environ.pop("RAI_PATH")


def test_redis_conf():
    config = Config()
    assert Path(config.database_conf).is_file()
    assert isinstance(config.database_conf, str)

    os.environ["REDIS_CONF"] = "not/a/path"
    config = Config()
    with pytest.raises(SSConfigError):
        config.database_conf
    os.environ.pop("REDIS_CONF")


def test_redis_exe():
    config = Config()
    assert Path(config.database_exe).is_file()
    assert isinstance(config.database_exe, str)

    os.environ["REDIS_PATH"] = "not/a/path"
    config = Config()
    with pytest.raises(SSConfigError):
        config.database_exe
    os.environ.pop("REDIS_PATH")


def test_redis_cli():
    config = Config()
    assert Path(config.redisai).is_file()
    assert isinstance(config.redisai, str)

    os.environ["REDIS_CLI_PATH"] = "not/a/path"
    config = Config()
    with pytest.raises(SSConfigError):
        config.database_cli
    os.environ.pop("REDIS_CLI_PATH")
