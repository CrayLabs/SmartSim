# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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
import os.path as osp
from pathlib import Path
from shutil import which

import psutil
import toml

from .error import SSConfigError


def expand_exe_path(exe):
    """Takes an executable and returns the full path to that executable

    :param exe: executable or file
    :type exe: str
    """

    # which returns none if not found
    in_path = which(exe)
    if not in_path:
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return os.path.abspath(exe)
        if os.path.isfile(exe) and not os.access(exe, os.X_OK):
            raise SSConfigError(f"File, {exe}, is not an executable")
        raise SSConfigError(f"Could not locate executable {exe}")
    return os.path.abspath(in_path)


class Config:
    def __init__(self):
        self.conf = self._load_conf()

    def _load_conf(self):
        def _load_from_home():
            try:
                home = os.environ["HOME"]
                config_path = osp.join(home, ".smartsim/config.toml")
                if osp.isfile(config_path):
                    return config_path
                return None
            except KeyError:
                return None

        def _load_from_sshome():
            try:
                home = os.environ["SMARTSIM_HOME"]
                config_path = osp.join(home, "config.toml")
                if osp.isfile(config_path):
                    return config_path
                config_path = osp.join(home, ".smartsim/config.toml")
                if osp.isfile(config_path):
                    return config_path
                return None
            except KeyError:
                return None

        def _load_defaults():
            package_path = os.path.dirname(os.path.abspath(__file__))
            lib_path = str(Path(package_path, "lib").resolve())
            bin_path = str(Path(package_path, "bin").resolve())
            conf_path = str(Path(package_path, "database", "redis6.conf"))

            if not osp.isdir(lib_path) and not osp.isdir(bin_path):
                msg = "SmartSim not installed with pre-built Redis libraries\n"
                msg += "Either install a pre-built wheel for SmartSim"
                msg += " or build SmartSim from source.\n"
                msg += " See documentation for instructions"
                raise SSConfigError(msg)

            default = {
                "redis": {
                    "bin": bin_path,
                    "config": conf_path,
                    "modules": {"ai": lib_path},
                },
                "smartsim": {"jm_interval": 15, "log_level": "info"},
                "test": {"launcher": "local", "device": "CPU", "interface": "ipogif0"},
            }
            return default

        conf_path = _load_from_home()
        if not conf_path:
            conf_path = _load_from_sshome()
        if not conf_path:
            return _load_defaults()
        config = toml.load(conf_path)
        return config

    @property
    def redisai(self):
        try:
            redisai = self.conf["redis"]["modules"]["ai"]
            if not osp.isfile(redisai):
                redisai = osp.join(redisai, "redisai.so")
                if not osp.isfile(redisai):
                    raise SSConfigError(
                        "RedisAI library path provided in SmartSim config could not be found"
                    )
            return redisai
        except KeyError:
            raise SSConfigError(
                "Could not find redis.modules.ai (path to redisai.so) in SmartSim config"
            )

    @property
    def redis_conf(self):
        redis_config = self.conf["redis"]["config"]
        if not osp.isfile(redis_config):
            smartsim_path = os.path.dirname(osp.abspath(__file__))
            conf_path = osp.join(smartsim_path, "database/redis6.conf")
            return conf_path
        return redis_config

    @property
    def redis_exe(self):
        try:
            redis_bin = self.conf["redis"]["bin"]
            redis_cli = osp.join(redis_bin, "redis-server")
            exe = expand_exe_path(redis_cli)
            return exe
        except KeyError:
            raise SSConfigError("Could not find redis.bin in SmartSim config")
        except SSConfigError as e:
            raise SSConfigError(
                "redis-server exe in SmartSim Config could not be used"
            ) from e

    @property
    def redis_cli(self):
        try:
            redis_bin = self.conf["redis"]["bin"]
            redis_cli = osp.join(redis_bin, "redis-cli")
            exe = expand_exe_path(redis_cli)
            return exe
        except KeyError:
            raise SSConfigError("Could not find redis.bin in SmartSim config")
        except SSConfigError as e:
            raise SSConfigError(
                "redis-cli executable in SmartSim Config could not be used"
            ) from e

    @property
    def test_launcher(self):
        try:
            if "SMARTSIM_TEST_LAUNCHER" in os.environ:
                return os.environ["SMARTSIM_TEST_LAUNCHER"]
            else:
                launcher = self.conf["test"]["launcher"]
                return launcher
        except KeyError:
            return "local"  # local by default

    @property
    def test_device(self):
        try:
            if "SMARTSIM_TEST_DEVICE" in os.environ:
                return os.environ["SMARTSIM_TEST_DEVICE"]
            else:
                device = self.conf["test"]["device"]
                return device
        except KeyError:
            return "CPU"  # cpu by default

    @property
    def test_interface(self):
        try:
            if "SMARTSIM_TEST_INTERFACE" in os.environ:
                return os.environ["SMARTSIM_TEST_INTERFACE"]
            else:
                interface = self.conf["test"]["interface"]
                return interface
        except KeyError:
            # try to pick a sensible one
            net_if_addrs = psutil.net_if_addrs()
            if "ipogif0" in net_if_addrs:
                return "ipogif0"
            elif "ib0" in net_if_addrs:
                return "ib0"
            # default to aries network
            return "ipogif0"

    @property
    def log_level(self):
        try:
            if "SMARTSIM_LOG_LEVEL" in os.environ:
                return os.environ["SMARTSIM_LOG_LEVEL"]
            else:
                level = self.conf["smartsim"]["log_level"]
                return level
        except KeyError:
            return "info"  # info by default

    @property
    def jm_interval(self):
        try:
            if "SMARTSIM_JM_INTERVAL" in os.environ:
                return int(os.environ["SMARTSIM_JM_INTERVAL"])
            else:
                num_seconds = self.conf["smartsim"]["jm_interval"]
                return int(num_seconds)
        except KeyError:
            return 15  # 15 seconds by default

    @property
    def test_account(self):
        try:
            if "SMARTSIM_TEST_ACCOUNT" in os.environ:
                return os.environ["SMARTSIM_TEST_ACCOUNT"]
            else:
                account = self.conf["test"]["account"]
                return account
        except KeyError:
            return ""  # no account by default


# initialize config instance
CONFIG = Config()
