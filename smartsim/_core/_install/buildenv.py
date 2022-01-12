import os
import platform
import site
import subprocess
import sys
import pprint
import pkg_resources

from pathlib import Path
from typing import Iterable
from pkg_resources import packaging

Version = packaging.version.Version
InvalidVersion = packaging.version.InvalidVersion


# NOTE: This will be imported by setup.py and hence no
#       smartsim related items or non-standand library
#       items should be imported here.

# TODO:
#  - check versions of prequisites
#  - include Ray versions


class SetupError(Exception):
    pass

# so as to not conflict with pkg_resources.packaging.version.Version
class Version_(str):
    def _convert_to_version(self, vers):
        if isinstance(vers, Version):
            return vers
        elif isinstance(vers, str):
            return Version(vers)
        elif isinstance(vers, Iterable):
            return Version('.'.join((str(item) for item in vers)))
        else:
            raise InvalidVersion(vers)

    @property
    def major(self):
        return Version(self).major

    @property
    def minor(self):
        return Version(self).minor

    @property
    def micro(self):
        return Version(self).micro

    def __gt__(self, cmp):
        try:
            return Version(self).__gt__(self._convert_to_version(cmp))
        except InvalidVersion:
            return super().__gt__(cmp)

    def __lt__(self, cmp):
        try:
            return Version(self).__lt__(self._convert_to_version(cmp))
        except InvalidVersion:
            return super().__lt__(cmp)

    def __eq__(self, cmp):
        try:
            return Version(self).__eq__(self._convert_to_version(cmp))
        except InvalidVersion:
            return super().__eq__(cmp)

    def __ge__(self, cmp):
        try:
            return Version(self).__ge__(self._convert_to_version(cmp))
        except InvalidVersion:
            return super().__ge__(cmp)

    def __le__(self, cmp):
        try:
            return Version(self).__le__(self._convert_to_version(cmp))
        except InvalidVersion:
            return super().__le__(cmp)


def get_env(var, default):
    return os.environ.get(var, default)

class RedisAIVersion(Version_):

    defaults = {
        "1.2.3": {
            "tensorflow": "2.4.2",
            "onnx": "1.7.0",
            "skl2onnx": "1.9.0",
            "onnxmltools": "1.7.0",
            "scikit-learn": "0.24.2",
            "torch": "1.7.1",
            "torchvision": "0.8.2"
        },
        "1.2.4": {
            "tensorflow": "2.4.2",
            "onnx": "1.7.0",
            "skl2onnx": "1.9.0",
            "onnxmltools": "1.7.0",
            "scikit-learn": "0.24.2",
            "torch": "1.7.1",
            "torchvision": "0.8.2"
        },
        "1.2.5": {
            "tensorflow": "2.6.2",
            "onnx": "1.9.0",
            "skl2onnx": "1.10.3",
            "onnxmltools": "1.10.0",
            "scikit-learn": "1.0.2",
            "torch": "1.9.1",
            "torchvision": "0.10.1"
        }
    }

    def __init__(self, vers):
        if vers not in self.defaults:
            raise SetupError(
                f"Invalid RedisAI version {vers}. Options are {self.defaults.keys()}")
        self.version = vers

    def __getattr__(self, name):
        return self.defaults[self.version][name]

    def get_defaults(self):
        return self.defaults[self.version]


class Versioner:
    """Buildtime configuration of third-party dependencies"""

    # compatible Python version
    PYTHON_MIN = Version_("3.7.0")

    # Versions
    SMARTSIM = Version_(get_env("SMARTSIM_VERSION", "0.3.2"))
    SMARTREDIS = Version_(get_env("SMARTREDIS_VERSION", "0.2.0"))
    SMARTSIM_SUFFIX = get_env("SMARTSIM_SUFFIX", "")

    # Redis
    REDIS = Version_(get_env("SMARTSIM_REDIS", "6.0.8"))
    REDIS_URL = get_env("SMARTSIM_REDIS_URL",
                        "https://github.com/redis/redis.git/")

    # RedisAI
    REDISAI = RedisAIVersion(get_env("SMARTSIM_REDISAI", "1.2.3"))
    REDISAI_URL = get_env("SMARTSIM_REDISAI_URL",
                          "https://github.com/RedisAI/RedisAI.git/")

    # ML/DL (based on RedisAI version defaults)
    TORCH = Version_(get_env("SMARTSIM_TORCH", REDISAI.torch))
    TORCHVISION = Version_(get_env("SMARTSIM_TORCHVIS", REDISAI.torchvision))
    TENSORFLOW = Version_(REDISAI.tensorflow)
    ONNX = Version_(REDISAI.onnx)

    def as_dict(self):
        packages = ["SMARTSIM", "SMARTREDIS", "REDIS",
                    "REDISAI", "TORCH", "TENSORFLOW", "ONNX"]
        versions = [self.SMARTSIM, self.SMARTREDIS, self.REDIS,
                    self.REDISAI, self.TORCH, self.TENSORFLOW, self.ONNX]
        vers = {"Packages": packages, "Versions": versions}
        return vers

    def ml_extras_required(self):
        """Optional ML/DL dependencies we suggest for the user.

        The defaults are based on the RedisAI version
        """
        ml_extras = []
        ml_defaults = self.REDISAI.get_defaults()

        # remove torch and torch vision as they will be installed
        # by the cli process for use in the RAI build. We don't install
        # them here as the user needs to decide between GPU/CPU. All other
        # libraries work on both devices
        del ml_defaults["torch"]
        del ml_defaults["torchvision"]

        for lib, vers in ml_defaults.items():
            ml_extras.append(f"{lib}=={vers}")
        return ml_extras

    def get_sha(self, setup_py_dir) -> str:
        """Get the git sha of the current branch"""
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=setup_py_dir)
                .decode("ascii")
                .strip()
            )
            return sha[:7]
        except Exception:
            return "Unknown"

    def write_version(self, setup_py_dir):
        """
        Write version info to version.py

        Use git_sha in the case where smartsim suffix is set in the environment
        """
        version = self.SMARTSIM
        if self.SMARTSIM_SUFFIX:
            git_sha = self.get_sha(setup_py_dir)
            version = f"{version}-{self.SMARTSIM_SUFFIX}-{git_sha}"

        version_file = setup_py_dir / "smartsim" / "version.py"
        with open(version_file, "w") as f:
            f.write("# This file is automatically generated by setup.py\n")
            f.write("# do not edit this file manually.\n\n")

            f.write(f"__version__ = '{version}'\n")
        return version


class BuildEnv:
    """Environment for building third-party dependencies"""

    # Compiler overrides
    CC = os.environ.get("CC", "gcc")
    CXX = os.environ.get("CXX", "g++")
    CFLAGS = os.environ.get("CFLAGS", "")
    CXXFLAGS = os.environ.get("CXXFLAGS", "")

    # build overrides
    MALLOC = os.environ.get("MALLOC", "libc")
    JOBS = os.environ.get("BUILD_JOBS", 1)

    # check for CC/GCC/ETC
    CHECKS = int(os.environ.get("NO_CHECKS", 0))
    PLATFORM = sys.platform

    def __init__(self):
        self.check_dependencies()

    def check_dependencies(self):
        deps = [
            "git",
            "git-lfs",
            "make",
            "wget",
            "cmake",
            self.CC,
            self.CXX
        ]
        if int(self.CHECKS) == 0:
            for dep in deps:
                self.check_build_dependency(dep)

    def __call__(self):
        # return the build env for the build process
        env = os.environ.copy()
        env.update({
            "CC": self.CC,
            "CXX": self.CXX,
            "CFLAGS": self.CFLAGS,
            "CXXFLAGS": self.CXXFLAGS
            })
        return env

    def as_dict(self):
        variables = ["CC", "CXX", "CFLAGS",
                     "CXXFLAGS", "MALLOC", "JOBS",
                     "PYTHON_VERSION", "PLATFORM"]
        values = [self.CC, self.CXX, self.CFLAGS,
                  self.CXXFLAGS, self.MALLOC, self.JOBS,
                  self.python_version, self.PLATFORM]
        env = {"Environment": variables, "Values": values}
        return env

    @property
    def python_version(self):
        return platform.python_version()

    def is_compatible_python(self, python_min):
        """Detect if system Python is too old"""
        sys_py = sys.version_info
        system_python = Version_(f"{sys_py.major}.{sys_py.minor}.{sys_py.micro}")
        if system_python < python_min:
            return False
        return True

    def is_windows(self):
        if self.PLATFORM in ["win32", "cygwin", "msys"]:
            return True
        return False

    def is_macos(self):
        if self.PLATFORM == "darwin":
            return True
        return False

    @property
    def site_packages_path(self):
        site_path = Path(site.getsitepackages()[0]).resolve()
        return site_path

    @property
    def torch_cmake_path(self):
        site_path = self.site_packages_path
        torch_path = site_path.joinpath("torch/share/cmake/Torch/").resolve()
        return str(torch_path)

    @staticmethod
    def get_cudnn_env():
        env = {
            "CUDNN_LIBRARY": os.environ.get("CUDNN_LIBRARY", None),
            "CUDNN_INCLUDE_DIR": os.environ.get("CUDNN_INCLUDE_DIR", None),
            "CUDNN_LIBRARY_PATH": os.environ.get("CUDNN_LIBRARY_PATH", None),
            "CUDNN_INCLUDE_PATH": os.environ.get("CUDNN_INCLUDE_PATH", None)
        }
        torch_cudnn_vars = ["CUDNN_LIBRARY", "CUDNN_INCLUDE_DIR"]
        caffe_cudnn_vars = ["CUDNN_INCLUDE_PATH", "CUDNN_LIBRARY_PATH"]

        torch_set = all([var in os.environ for var in torch_cudnn_vars])
        caffe_set = all([var in os.environ for var in caffe_cudnn_vars])

        # check for both sets, if only one exists, set the other
        # this handles older versions which use different env vars
        if not torch_set and not caffe_set:
            return None # return None as we don't want to raise a warning here
        if torch_set and not caffe_set:
            env["CUDNN_INCLUDE_PATH"] = env["CUDNN_INCLUDE_DIR"]
            env["CUDNN_LIBRARY_PATH"] = env["CUDNN_LIBRARY"]
        elif caffe_set and not torch_set:
            env["CUDNN_INCLUDE_DIR"] = env["CUDNN_INCLUDE_PATH"]
            env["CUDNN_LIBRARY"] = env["CUDNN_LIBRARY_PATH"]
        return env

    def check_build_dependency(self, command):
        # TODO expand this to parse and check versions.
        try:
            out = subprocess.check_call([command, "--version"],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        except OSError:
            raise SetupError(f"{command} must be installed to build SmartSim") from None

    @staticmethod
    def check_installed(package, version=None):
        """Check if a package is installed. If version is provided, check if
        it's a compatible version. (major and minor the same)"""
        try:
            installed = Version_(pkg_resources.get_distribution(package).version)
            if version:
                # detect if major or minor versions differ
                if installed.major != version.major or installed.minor != version.minor:
                    msg = (f"Incompatible version for {package} detected.\n" +
                          f"{package} {version} requested but {package} {installed} installed.")
                    raise SetupError(msg)
            return True
        except pkg_resources.DistributionNotFound:
            return False
