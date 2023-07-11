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
import psutil
import typing as t

from functools import lru_cache
from pathlib import Path

from ...error import SSConfigError
from ..utils.helpers import expand_exe_path

# Configuration Values
#
# These values can be set through environment variables to
# override the default behavior of SmartSim.
#
# RAI_PATH
#   - Path to the RAI shared library
#   - Default: /smartsim/smartsim/_core/lib/redisai.so
#
# REDIS_CONF
#   - Path to the redis.conf file
#   - Default: /SmartSim/smartsim/_core/config/redis.conf
#
# REDIS_PATH
#   - Path to the redis-server executable
#   - Default: /SmartSim/smartsim/_core/bin/redis-server
#
# REDIS_CLI_PATH
#   - Path to the redis-cli executable
#   - Default: /SmartSim/smartsim/_core/bin/redis-cli
#
# SMARTSIM_LOG_LEVEL
#   - Log level for SmartSim
#   - Default: info
#
# SMARTSIM_JM_INTERVAL
#   - polling interval for communication with scheduler
#   - default: 10 seconds
#


# Testing Configuration Values
#
# SMARTSIM_TEST_INTERFACE
#  - Network interface to use for testing
#  - Default: auto-detected
#
# SMARTSIM_TEST_LAUNCHER
#  - type of launcher to use for testing
#  - Default: Local
#
# SMARTSIM_TEST_DEVICE
#  - CPU or GPU for model serving tests
#  - Default: CPU
#
# SMARTSIM_TEST_NUM_GPUS
#  - Number of GPUs on the host for testing
#  - Defaults: 1
#
# SMARTSIM_TEST_ACCOUNT
#  - Account used to run full launcher test suite on external systems
#  - Default: None


class Config:
    def __init__(self) -> None:
        # SmartSim/smartsim/_core
        self.core_path = Path(os.path.abspath(__file__)).parent.parent

        dependency_path = os.environ.get("SMARTSIM_DEP_INSTALL_PATH", self.core_path)

        self.lib_path = Path(dependency_path, "lib").resolve()
        self.bin_path = Path(dependency_path, "bin").resolve()
        self.conf_path = Path(dependency_path, "config", "redis.conf")

    @property
    def redisai(self) -> str:
        rai_path = self.lib_path / "redisai.so"
        redisai = Path(os.environ.get("RAI_PATH", rai_path)).resolve()
        if not redisai.is_file():
            raise SSConfigError(
                "RedisAI dependency not found. Build with `smart` cli "
                "or specify RAI_PATH"
            )
        return str(redisai)

    @property
    def database_conf(self) -> str:
        conf = Path(os.environ.get("REDIS_CONF", self.conf_path)).resolve()
        if not conf.is_file():
            raise SSConfigError(
                "Database configuration file at REDIS_CONF could not be found"
            )
        return str(conf)

    @property
    def database_exe(self) -> str:
        try:
            database_exe = next(self.bin_path.glob("*-server"))
            database = Path(os.environ.get("REDIS_PATH", database_exe)).resolve()
            exe = expand_exe_path(str(database))
            return exe
        except (TypeError, FileNotFoundError) as e:
            raise SSConfigError(
                "Specified database binary at REDIS_PATH could not be used"
            ) from e

    @property
    def database_cli(self) -> str:
        try:
            redis_cli_exe = next(self.bin_path.glob("*-cli"))
            redis_cli = Path(os.environ.get("REDIS_CLI_PATH", redis_cli_exe)).resolve()
            exe = expand_exe_path(str(redis_cli))
            return exe
        except (TypeError, FileNotFoundError) as e:
            raise SSConfigError(
                "Specified Redis binary at REDIS_CLI_PATH could not be used"
            ) from e

    @property
    def log_level(self) -> str:
        return os.environ.get("SMARTSIM_LOG_LEVEL", "info")

    @property
    def jm_interval(self) -> int:
        return int(os.environ.get("SMARTSIM_JM_INTERVAL") or 10)

    @property
    def wlm_trials(self) -> int:
        return int(os.environ.get("SMARTSIM_WLM_TRIALS") or 10)

    @property
    def test_launcher(self) -> str:  # pragma: no cover
        return os.environ.get("SMARTSIM_TEST_LAUNCHER", "local")

    @property
    def test_device(self) -> str:  # pragma: no cover
        return os.environ.get("SMARTSIM_TEST_DEVICE", "CPU")

    @property
    def test_num_gpus(self) -> int:  # pragma: no cover
        return int(os.environ.get("SMARTSIM_TEST_NUM_GPUS") or 1)

    @property
    def test_port(self) -> int:  # pragma: no cover
        return int(os.environ.get("SMARTSIM_TEST_PORT", 6780))

    @property
    def test_interface(self) -> t.List[str]:  # pragma: no cover
        if interfaces_cfg := os.environ.get("SMARTSIM_TEST_INTERFACE", None):
            return interfaces_cfg.split(",")

        # try to pick a sensible one
        net_if_addrs = psutil.net_if_addrs()
        if "ipogif0" in net_if_addrs:
            return ["ipogif0"]
        if "hsn0" in net_if_addrs:
            return [
                net_if_addr
                for net_if_addr in net_if_addrs
                if net_if_addr.startswith("hsn")
            ]
        if "ib0" in net_if_addrs:
            return ["ib0"]
        # default to aries network
        return ["lo"]

    @property
    def test_account(self) -> t.Optional[str]:  # pragma: no cover
        # no account by default
        return os.environ.get("SMARTSIM_TEST_ACCOUNT", None)


@lru_cache(maxsize=128, typed=False)
def get_config() -> Config:
    # wrap into a function with a cached result
    return Config()
