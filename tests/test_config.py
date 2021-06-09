import os

import pytest

from smartsim.config import Config
from smartsim.error import SSConfigError


def test_redisai():
    config = Config()

    config.conf["redis"]["modules"]["ai"] = "not/a/path"
    with pytest.raises(SSConfigError):
        config.redisai

    config.conf.pop("redis")
    with pytest.raises(SSConfigError):
        config.redisai


def test_redisip():
    config = Config()

    config.conf["redis"]["modules"]["ip"] = "not/a/path"
    with pytest.raises(SSConfigError):
        config.redisip

    config.conf.pop("redis")
    with pytest.raises(SSConfigError):
        config.redisip


def test_redis_conf():
    config = Config()
    config.conf["redis"]["config"] = "not/a/path"
    assert os.path.isfile(config.redis_conf)


def test_redis_exe():
    config = Config()
    config.conf["redis"]["bin"] = "not/a/path"
    with pytest.raises(SSConfigError):
        config.redis_exe
    config.conf.pop("redis")
    with pytest.raises(SSConfigError):
        config.redis_exe


def test_redis_cli():
    config = Config()
    config.redis_cli
    config.conf["redis"]["bin"] = "not/a/path"
    with pytest.raises(SSConfigError):
        config.redis_cli
    config.conf.pop("redis")
    with pytest.raises(SSConfigError):
        config.redis_cli


def test_launcher_log_interval_attributes():
    config = Config()
    defaults = ["local", "info", "15"]
    environ_keys = [
        "SMARTSIM_TEST_LAUNCHER",
        "SMARTSIM_LOG_LEVEL",
        "SMARTSIM_JM_INTERVAL",
    ]
    environ_vals = [os.environ.get(val, None) for val in environ_keys]

    for i, key in enumerate(environ_keys):
        os.environ[key] = defaults[i]
    assert config.test_launcher == defaults[0]
    assert config.log_level == defaults[1]
    assert config.jm_interval == int(defaults[2])

    for key in environ_keys:  # test the KeyError exceptions
        os.environ.pop(key)
    config.conf.pop("test")
    config.conf.pop("smartsim")
    assert config.test_launcher == defaults[0]
    assert config.log_level == defaults[1]
    assert config.jm_interval == int(defaults[2])

    for i in range(
        len(environ_vals)
    ):  # set environ key-val pairs back to their original state
        if environ_vals[i]:
            os.environ[environ_keys[i]] = environ_vals[i]
