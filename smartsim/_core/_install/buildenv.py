import os
import platform
import site
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pkg_resources
from pkg_resources import packaging

Version = packaging.version.Version
InvalidVersion = packaging.version.InvalidVersion


# NOTE: This will be imported by setup.py and hence no
#       smartsim related items or non-standand library
#       items should be imported here.


class SetupError(Exception):
    """A simple exception class for errors in _install.buildenv file.
    This is primarily used to interrupt the setup.py build in case
    of a failure, and is caught frequently in the CLI which attempts
    to suppress and log the errors thrown here.

    One note is that this error must be imported from here to be
    caught and not defined anywhere else. Since we use this class
    in the installation, having a separate error file would mean
    another module to manually import into setup.py

    See setup.py for more
    """

    pass


# so as to not conflict with pkg_resources.packaging.version.Version
class Version_(str):
    """A subclass of pkg_resources.packaging.version.Version that
    includes some helper methods for comparing versions.
    """

    def _convert_to_version(self, vers):
        if isinstance(vers, Version):
            return vers
        elif isinstance(vers, str):
            return Version(vers)
        elif isinstance(vers, Iterable):
            return Version(".".join((str(item) for item in vers)))
        else:
            raise InvalidVersion(vers)

    @property
    def major(self):
        # Version(self).major doesn't work for all Python distributions
        # see https://github.com/lebedov/python-pdfbox/issues/28
        return int(pkg_resources.parse_version(self).base_version.split(".")[0])

    @property
    def minor(self):
        return int(pkg_resources.parse_version(self).base_version.split(".")[1])

    @property
    def micro(self):
        return int(pkg_resources.parse_version(self).base_version.split(".")[2])

    @property
    def patch(self):
        # return micro with string modifier i.e. 1.2.3+cpu -> 3+cpu
        return str(pkg_resources.parse_version(self)).split(".")[2]

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
    """A subclass of Version_ that holds the dependency sets for RedisAI

    this class serves two purposes:

    1. It is used to populate the [ml] ``extras_require`` of the setup.py.
    This is because the RedisAI version will determine which ML based
    dependencies are required.

    2. Used to set the default values for PyTorch, TF, and ONNX
    given the SMARTSIM_REDISAI env var set by the user.
    """

    defaults = {
        "1.2.3": {
            "tensorflow": "2.5.2",
            "onnx": "1.9.0",
            "skl2onnx": "1.10.3",
            "onnxmltools": "1.10.0",
            "scikit-learn": "1.0.2",
            "torch": "1.7.1",
            "torchvision": "0.8.2",
        },
        "1.2.5": {
            "tensorflow": "2.6.2",
            "onnx": "1.9.0",
            "skl2onnx": "1.10.3",
            "onnxmltools": "1.10.0",
            "scikit-learn": "1.0.2",
            "torch": "1.9.1",
            "torchvision": "0.10.1",
        },
    }
    # deps are the same between the following versions
    defaults["1.2.4"] = defaults["1.2.3"]

    def __init__(self, vers):
        if vers not in self.defaults:
            if vers.startswith("1.2"):
                # resolve to latest version for 1.2.x
                # the str representation will still be 1.2.x
                self.version = "1.2.5"
            else:
                raise SetupError(
                    f"Invalid RedisAI version {vers}. Options are {self.defaults.keys()}"
                )
        else:
            self.version = vers

    def __getattr__(self, name):
        return self.defaults[self.version][name]

    def get_defaults(self):
        return self.defaults[self.version]


class Versioner:
    """Versioner is responsible fo managing all the versions
    within SmartSim including SmartSim itself.

    SmartSim's version is written into version.py upon pip install
    by using this class in setup.py. By setting SMARTSIM_SUFFIX,
    the version will be written as a "dirty" version with the
    git-sha appended.
    i.e.
        export SMARTSIM_SUFFIX=nightly
        pip install -e .
        smartsim.__version__ == 0.3.2+nightly-3fe23ff

    Versioner manages third party dependencies and their versions
    as well. These versions are used by the ``smart`` cli in the
    ``smart build`` command to determine which dependency versions
    to look for and download.
    """

    # compatible Python version
    PYTHON_MIN = Version_("3.7.0")

    # Versions
    SMARTSIM = Version_(get_env("SMARTSIM_VERSION", "0.4.0"))
    SMARTREDIS = Version_(get_env("SMARTREDIS_VERSION", "0.3.0"))
    SMARTSIM_SUFFIX = get_env("SMARTSIM_SUFFIX", "")

    # Redis
    REDIS = Version_(get_env("SMARTSIM_REDIS", "6.0.8"))
    REDIS_URL = get_env("SMARTSIM_REDIS_URL", "https://github.com/redis/redis.git/")
    REDIS_BRANCH = get_env("SMARTSIM_REDIS_BRANCH", REDIS)

    # RedisAI
    REDISAI = RedisAIVersion(get_env("SMARTSIM_REDISAI", "1.2.3"))
    REDISAI_URL = get_env(
        "SMARTSIM_REDISAI_URL", "https://github.com/RedisAI/RedisAI.git/"
    )
    REDISAI_BRANCH = get_env("SMARTSIM_REDISAI_BRANCH", f"v{REDISAI}")

    # ML/DL (based on RedisAI version defaults)
    # torch can be set by the user because we download that for them
    TORCH = Version_(get_env("SMARTSIM_TORCH", REDISAI.torch))
    TORCHVISION = Version_(get_env("SMARTSIM_TORCHVIS", REDISAI.torchvision))

    # TensorFlow and ONNX only use the defaults, but these are not built into
    # the RedisAI package and therefore the user is free to pick other versions.
    TENSORFLOW = Version_(REDISAI.tensorflow)
    ONNX = Version_(REDISAI.onnx)

    def as_dict(self):
        packages = [
            "SMARTSIM",
            "SMARTREDIS",
            "REDIS",
            "REDISAI",
            "TORCH",
            "TENSORFLOW",
            "ONNX",
        ]
        versions = [
            self.SMARTSIM,
            self.SMARTREDIS,
            self.REDIS,
            self.REDISAI,
            self.TORCH,
            self.TENSORFLOW,
            self.ONNX,
        ]
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
            # return empty string if not in git-repo
            return ""

    def write_version(self, setup_py_dir):
        """
        Write version info to version.py

        Use git_sha in the case where smartsim suffix is set in the environment
        """
        version = self.SMARTSIM
        if self.SMARTSIM_SUFFIX:
            git_sha = self.get_sha(setup_py_dir)
            if git_sha:
                version = f"{version}+{self.SMARTSIM_SUFFIX}.{git_sha}"
            else:
                # wheel build (python -m build) shouldn't include git sha
                version = f"{version}+{self.SMARTSIM_SUFFIX}"

        version_file = setup_py_dir / "smartsim" / "version.py"
        with open(version_file, "w") as f:
            f.write("# This file is automatically generated by setup.py\n")
            f.write("# do not edit this file manually.\n\n")

            f.write(f"__version__ = '{version}'\n")
        return version


class BuildEnv:
    """Environment for building third-party dependencies

    BuildEnv provides a method for configuring how the third-party
    dependencies within SmartSim are built, namely Redis/KeyDB
    and RedisAI.

    The environment variables listed here can be set to control the
    Redis build in the pip wheel build as well as the Redis and RedisAI
    build executed by the CLI.

    Build tools are also checked for here and if they are not found
    then a SetupError is raised.

    Adding the -v flag to ``smart build`` will show the build environment
    being used.
    """

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

    def __init__(self, checks=True):
        if checks:
            self.check_dependencies()

    def check_dependencies(self):
        deps = ["git", "git-lfs", "make", "wget", "cmake", self.CC, self.CXX]
        if int(self.CHECKS) == 0:
            for dep in deps:
                self.check_build_dependency(dep)

    def __call__(self):
        # return the build env for the build process
        env = os.environ.copy()
        env.update(
            {
                "CC": self.CC,
                "CXX": self.CXX,
                "CFLAGS": self.CFLAGS,
                "CXXFLAGS": self.CXXFLAGS,
            }
        )
        return env

    def as_dict(self):
        variables = [
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "MALLOC",
            "JOBS",
            "PYTHON_VERSION",
            "PLATFORM",
        ]
        values = [
            self.CC,
            self.CXX,
            self.CFLAGS,
            self.CXXFLAGS,
            self.MALLOC,
            self.JOBS,
            self.python_version,
            self.PLATFORM,
        ]
        env = {"Environment": variables, "Values": values}
        return env

    @property
    def python_version(self):
        return platform.python_version()

    def is_compatible_python(self, python_min):
        """Detect if system Python is too old"""
        sys_py = sys.version_info
        system_python = Version_(f"{sys_py.major}.{sys_py.minor}.{sys_py.micro}")
        return system_python > python_min

    def is_windows(self):
        return self.PLATFORM in ["win32", "cygwin", "msys"]

    def is_macos(self):
        return self.PLATFORM == "darwin"

    @property
    def torch_cmake_path(self):
        """Find the path to the cmake directory within a
        pip installed pytorch package"""

        def _torch_import_path():
            """Find through importing torch"""
            try:
                import torch as t

                torch_paths = [Path(p) for p in t.__path__]
                for _path in torch_paths:
                    torch_path = _path / "share/cmake/Torch"
                    if torch_path.is_dir():
                        return torch_path
                return None
            except ModuleNotFoundError:
                return None

        def _torch_site_path():
            """find torch through site packages"""
            site_paths = [Path(p) for p in site.getsitepackages()]

            # check user site (~/.local/lib)
            if Path(site.USER_SITE).is_dir():
                site_paths.append(Path(site.USER_SITE))

            for _path in site_paths:
                torch_path = _path / "torch/share/cmake/Torch"
                if torch_path.is_dir():
                    return torch_path
            return None

        torch_path = _torch_import_path()
        if not torch_path:
            torch_path = _torch_site_path()
        if not torch_path:
            raise SetupError("Could not locate torch cmake path")
        return str(torch_path)

    @staticmethod
    def get_cudnn_env():
        """Collect the environment variables needed for Caffe (Pytorch)
        and throw an error if they are not found

        Specifically make sure to set at least one set of:
            - CUDNN_LIBRARY and CUDNN_INCLUDE_DIR
                or
            - CUDNN_LIBRARY_PATH and CUDNN_INCLUDE_PATH
        """
        env = {
            "CUDNN_LIBRARY": os.environ.get("CUDNN_LIBRARY", None),
            "CUDNN_INCLUDE_DIR": os.environ.get("CUDNN_INCLUDE_DIR", None),
            "CUDNN_LIBRARY_PATH": os.environ.get("CUDNN_LIBRARY_PATH", None),
            "CUDNN_INCLUDE_PATH": os.environ.get("CUDNN_INCLUDE_PATH", None),
        }
        torch_cudnn_vars = ["CUDNN_LIBRARY", "CUDNN_INCLUDE_DIR"]
        caffe_cudnn_vars = ["CUDNN_INCLUDE_PATH", "CUDNN_LIBRARY_PATH"]

        torch_set = all([var in os.environ for var in torch_cudnn_vars])
        caffe_set = all([var in os.environ for var in caffe_cudnn_vars])

        # check for both sets, if only one exists, set the other
        # this handles older versions which use different env vars
        if not torch_set and not caffe_set:
            return None  # return None as we don't want to raise a warning here
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
            out = subprocess.check_call(
                [command, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
                    msg = (
                        f"Incompatible version for {package} detected.\n"
                        + f"{package} {version} requested but {package} {installed} installed."
                    )
                    raise SetupError(msg)
            return True
        except pkg_resources.DistributionNotFound:
            return False
