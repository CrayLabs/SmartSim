
import os
import sys
import toml
import os.path as osp

from .error import SSConfigError
from .utils.helpers import expand_exe_path

class Config:

    def __init__(self):
        self.conf = self._load_conf()

    def _load_conf(self):
        try:
            home = os.environ["HOME"]
            config_path = osp.join(home, ".smartsim/config.toml")
            config = toml.load(config_path)
            return config
        except FileNotFoundError:
            raise SSConfigError(
                f"Could not find SmartSim config. Looked at {config_path}"
            )

    @property
    def redisai(self):
        try:
            redisai = self.conf["redis"]["ai"]
            device = redisai["device"]
            lib = redisai["install_path"]
            lib_path = osp.join(lib, "redisai.so")
            if not osp.isfile(lib_path):
                raise SSConfigError("RedisAI library path provided in SmartSim config could be found")
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
                raise SSConfigError("RedisIP library path provided in SmartSim config could be found")
            return lib_path
        except KeyError:
            raise SSConfigError("Could not find redis.ip in SmartSim config")

    @property
    def redis_conf(self):
        try:
            redis = self.conf["redis"]
            conf_path = redis["config"]
            if not osp.isfile(conf_path):
                raise SSConfigError("Redis 'config' provided in SmartSim config could be found")
            return conf_path
        except KeyError:
            smartsim_path = os.path.dirname(os.path.abspath(__file__))
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
            lib_path = redis["cli"]
            exe_path = osp.join(lib_path, "redis-cli")
            exe = expand_exe_path(exe_path)
            return exe
        except KeyError:
            raise SSConfigError("Could not find redis 'cli' executable in SmartSim config")
        except SSConfigError as e:
            raise SSConfigError("Redis-cli executable in SmartSim Config could not be used") from e

    @property
    def test_launcher(self):
        try:
            if "SMARTSIM_TEST_LAUNCHER" in os.environ:
                return os.environ["SMARTSIM_TEST_LAUNCHER"]
            else:
                launcher = self.conf["test"]["launcher"]
                return launcher
        except KeyError:
            return "local" # local by default

    @property
    def log_level(self):
        try:
            if "SMARTSIM_LOG_LEVEL" in os.environ:
                return os.environ["SMARTSIM_LOG_LEVEL"]
            else:
                level = self.conf["smartsim"]["log_level"]
                return level
        except KeyError:
            return "info" # local by default

    @property
    def jm_interval(self):
        try:
            if "SMARTSIM_JM_INTERVAL" in os.environ:
                return os.environ["SMARTSIM_JM_INTERVAL"]
            else:
                num_seconds = self.conf["smartsim"]["jm_interval"]
                return int(num_seconds)
        except KeyError:
            return 15 # 15 seconds by default


# initialize config instance
CONFIG = Config()
