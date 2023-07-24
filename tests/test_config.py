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
import shutil

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

    os.environ["RAI_PATH"] = "not/an/so/file"
    config = Config()
    with pytest.raises(SSConfigError):
        config.redisai
    os.environ.pop("RAI_PATH")


def test_redisai_invalid_rai_path(fileutils, monkeypatch):
    """Ensure that looking for redisai module with invalid RAI_PATH fails"""
    test_dir = fileutils.make_test_dir()
    lib_dir = Path(f"{test_dir}/lib")
    rai_file_path = lib_dir / "redisai.so"

    env = os.environ.copy()
    env["RAI_PATH"] = str(rai_file_path)
    monkeypatch.setattr(os, "environ", env)

    config = Config()

    # Fail when no file exists
    with pytest.raises(SSConfigError) as ex:
        _ = config.redisai

    assert 'RedisAI dependency not found' in ex.value.args[0]


def test_redisai_valid_rai_path(fileutils, monkeypatch):
    """Ensure that looking for redisai module with RAI_PATH set works"""
    test_dir = fileutils.make_test_dir()
    lib_dir = Path(f"{test_dir}/lib")
    rai_file_path = lib_dir / "redisai.so"

    env = os.environ.copy()
    env["RAI_PATH"] = str(rai_file_path)
    monkeypatch.setattr(os, "environ", env)

    if lib_dir.exists():
        shutil.rmtree(test_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Add a file matching RAI_PATH and ensure it is found
    with open(rai_file_path, "w+") as f:
        f.write("mock module...")
    
    config = Config()
    assert config.redisai
    assert Path(config.redisai).is_file()
    assert isinstance(config.redisai, str)

    shutil.rmtree(test_dir)


def test_redisai_invalid_lib_path(fileutils, monkeypatch):
    """Ensure that looking for redisai module with both RAI_PATH and lib_path NOT set fails"""
    test_dir = fileutils.make_test_dir()
    lib_dir = Path(f"{test_dir}/lib")

    default_install_dir = Path(f"{test_dir}/defaults")
    default_lib_dir = default_install_dir / "lib"

    env = os.environ.copy()
    env["SMARTSIM_DEP_INSTALL_PATH"] = str(default_install_dir)

    monkeypatch.setattr(os, "environ", env)

    if lib_dir.exists():
        shutil.rmtree(test_dir)

    if default_lib_dir.exists():
        shutil.rmtree(default_lib_dir)
    
    config = Config()
    # Fail when no file exists
    with pytest.raises(SSConfigError) as ex:
        _ = config.redisai

    assert 'RedisAI dependency not found' in ex.value.args[0]


def test_redisai_valid_lib_path(fileutils, monkeypatch):
    """Ensure that looking for redisai module with RAI_PATH NOT set works"""
    test_dir = fileutils.make_test_dir()
    lib_dir = Path(f"{test_dir}/lib")

    default_install_dir = Path(f"{test_dir}/defaults")
    default_lib_dir = default_install_dir / "lib"
    rai_def_lib_path = default_lib_dir / "redisai.so"

    env = os.environ.copy()
    env["SMARTSIM_DEP_INSTALL_PATH"] = str(default_install_dir)

    monkeypatch.setattr(os, "environ", env)

    if lib_dir.exists():
        shutil.rmtree(test_dir)

    if default_lib_dir.exists():
        shutil.rmtree(default_lib_dir)
    default_lib_dir.mkdir(parents=True, exist_ok=True)

    # Add a file matching RAI_PATH and ensure it is found
    with open(rai_def_lib_path, "w+") as f:
        f.write("mock default module...")
    
    config = Config()
    assert config.redisai
    assert Path(config.redisai).is_file()
    assert isinstance(config.redisai, str)

    shutil.rmtree(test_dir)


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
