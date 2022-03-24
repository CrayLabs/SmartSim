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
