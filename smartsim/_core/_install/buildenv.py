
import os
import sys
import site
import subprocess
import platform
import pkg_resources
from pathlib import Path

# NOTE: This will be imported by setup.py and hence no
#       smartsim related items or non-standand library
#       items should be imported here.

# TODO:
#  - check versions of prequisites
#  - check urls
#  - ensure PyTorch versions will work for RedisAI version (should support a range)
#  - Add support for RedisAI 1.2.5


class SetupError(Exception):
    pass

class Versioner:

    # compatible Python version
    PYTHON_MIN = (3, 7, 0)

    # Versions
    SMARTSIM = os.environ.get("SMARTSIM_VERSION", "0.3.2")
    SMARTSIM_SUFFIX = os.environ.get("SMARTSIM_SUFFIX", "")
    SMARTREDIS = os.environ.get("SMARTREDIS_VERSION", "0.2.0")

    # Redis
    REDIS = os.environ.get("SMARTSIM_REDIS", "6.0.8")
    REDIS_URL = os.environ.get("SMARTSIM_REDIS_URL", "https://github.com/redis/redis.git/")

    # RedisAI
    REDISAI = os.environ.get("SMARTSIM_REDISAI", "1.2.3")
    REDISAI_URL = os.environ.get("SMARTSIM_REDISAI_URL", "https://github.com/RedisAI/RedisAI.git/")

    # TORCH
    TORCH = os.environ.get("SMARTSIM_TORCH", "1.7.1")
    TORCHVISION = os.environ.get("SMARTSIM_TORCHVIS", "0.8.2")


    def __init__(self):
        # align RedisAI versions with ML packages
        if self.REDISAI == "1.2.3":
            self.TENSORFLOW = "2.4.2"
            self.ONNX = "1.7.0"
            self.SKL2ONNX = "1.9.0"
            self.ONNXML = "1.7.0"
        else:
            raise SetupError("Unsupported version of RedisAI: {}".format(self.REDISAI))

    def get_sha(self, setup_py_dir) -> str:
        try:
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=setup_py_dir).decode('ascii').strip()
            return sha[:7]
        except Exception:
            return 'Unknown'

    def onnx_packages(self):
        return [
            f"skl2onnx=={self.SKL2ONNX}",
            f"onnx=={self.ONNX}",
            f"onnxmltools=={self.ONNXML}"
            ]

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

    # environment overrides
    CC = os.environ.get('CC', "gcc")
    CXX = os.environ.get('CXX', "g++")
    MALLOC = os.environ.get("MALLOC", "libc")
    JOBS = os.environ.get("BUILD_JOBS", None)

    # check for CC/GCC/ETC
    CHECKS = int(os.environ.get("NO_CHECKS", 0))
    PLATFORM = sys.platform

    def __init__(self):
        self.check_dependencies()

    def check_dependencies(self):
        if int(self.CHECKS) != 0:
            self.check_prereq("git")
            self.check_prereq(self.CC)
            self.check_prereq(self.CXX)
            self.check_prereq("make")

    def __call__(self):
        # return the build env for the build process
        return {
            "CC": self.CC,
            "CXX": self.CXX,
            "CFLAGS": self.CFLAGS
        }

    @property
    def python_version(self):
        return platform.python_version()

    def is_compatible_python(self, python_min):
        if sys.version_info < python_min:
            return False
        return True

    def is_windows(self):
        if self.PLATFORM in ['win32', 'cygwin', "msys"]:
            return True
        return False

    def is_macos(self):
        if self.PLATFORM == 'darwin':
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

    def check_prereq(self, command):
        try:
            out = subprocess.check_output([command, '--version'])
        except OSError as e:
            raise RuntimeError(
                f"{command} must be installed to build SmartSim") from e

    @staticmethod
    def check_installed(package):
        try:
            pkg_resources.get_distribution(package)
            return True
        except pkg_resources.DistributionNotFound:
            return False

    @staticmethod
    def check_installed(package, version=None):
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if version:
                installed_major, installed_minor, _ = installed_version.split(".")
                supported_major, supported_minor, _ = version.split(".")

                if int(installed_major) != int(supported_major) \
                    or int(installed_minor) != int(supported_minor):
                        msg = f"Incompatible version for {package} detected.\n"
                        msg = f"{package} {version} requested but {package} {installed_version}"
                        raise SetupError(msg)
            return True
        except pkg_resources.DistributionNotFound:
            return False
