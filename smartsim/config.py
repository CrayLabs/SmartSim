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
import sys
from shutil import which

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

        conf_path = _load_from_home()
        if not conf_path:
            conf_path = _load_from_sshome()
        if not conf_path:
            msg = "Could not find SmartSim config\n"
            msg += "This usually means you haven't written a config.toml for SmartSim\n"
            msg += "Template Config\n"
            msg += _template_config
            msg += "\nCopy paste this in $HOME/.smartsim/config.toml"
            msg += " and replace the /path/to/.. with paths to each of the libraries\n"
            msg += "Optionally, you can also put the config in $SMARTSIM_HOME\n"
            raise SSConfigError(msg)

        config = toml.load(conf_path)
        return config

    @property
    def redisai(self):
        try:
            redisai = self.conf["redis"]["ai"]
            device = redisai["device"]
            lib = redisai["install_path"]
            lib_path = osp.join(lib, "redisai.so")
            if not osp.isfile(lib_path):
                raise SSConfigError(
                    "RedisAI library path provided in SmartSim config could not be found"
                )
            return lib_path, device
        except KeyError:
            raise SSConfigError("Could not find redis.ai in SmartSim config")

    @property
    def redisip(self):
        try:
            redisip = self.conf["redis"]["ip"]
            lib = redisip["install_path"]
            suffix = ".dylib" if sys.platform == "darwin" else ".so"
            lib_path = osp.join(lib, "libredisip" + suffix)
            if not osp.isfile(lib_path):
                raise SSConfigError(
                    "RedisIP library path provided in SmartSim config could not be found"
                )
            return lib_path
        except KeyError:
            raise SSConfigError("Could not find redis.ip in SmartSim config")

    @property
    def redis_conf(self):
        try:
            redis = self.conf["redis"]
            conf_path = redis["config"]
            if not osp.isfile(conf_path):
                raise SSConfigError(
                    "Redis 'config' provided in SmartSim config could not be found"
                )
            return conf_path
        except KeyError:
            smartsim_path = os.path.dirname(osp.abspath(__file__))
            conf_path = osp.join(smartsim_path, "database/redis6.conf")
            return conf_path

    @property
    def redis_exe(self):
        try:
            redis = self.conf["redis"]
            exe_path = redis["exe"]
            exe = expand_exe_path(exe_path)
            return exe
        except KeyError:
            raise SSConfigError("Could not find redis `exe` in SmartSim config")
        except SSConfigError as e:
            raise SSConfigError("Redis exe in SmartSim Config could not be used") from e

    @property
    def redis_cli(self):
        try:
            redis = self.conf["redis"]
            exe_path = redis["cli"]
            exe = expand_exe_path(exe_path)
            return exe
        except KeyError:
            raise SSConfigError(
                "Could not find redis 'cli' executable in SmartSim config"
            )
        except SSConfigError as e:
            raise SSConfigError(
                "Redis-cli executable in SmartSim Config could not be used"
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
                return os.environ["SMARTSIM_JM_INTERVAL"]
            else:
                num_seconds = self.conf["smartsim"]["jm_interval"]
                return int(num_seconds)
        except KeyError:
            return 15  # 15 seconds by default


_template_config = """
[smartsim]
# number of seconds per job status update
# for jobs on WLM system (e.g. slurm, pbs, etc)
jm_interval = 15    # default
log_level = "info" # default

[redis]
# path to where "redis-server" and "redis-cli" binaries are located
exe = "/path/to/redis/src/redis-server"
config = "/path/to/redis.conf" # optional!
cli = "/path/to/redis/src/redis-cli"

  [redis.ai]
  # path to the redisai "install_cpu" or "install_gpu" dir
  device = "cpu" # cpu or gpu
  install_path = "/path/to/RedisAI/install-cpu/"

  [redis.ip]
  # path to build dir for RedisIP
  install_path = "/path/to/RedisIP/build/"

[test]
launcher = "local" # default
"""


# initialize config instance
CONFIG = Config()
