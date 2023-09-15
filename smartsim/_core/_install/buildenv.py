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

# pylint: disable=invalid-name

import importlib.metadata
import os
import platform
import subprocess
import sys
import typing as t
from pathlib import Path
from typing import Iterable

# NOTE: This will be imported by setup.py and hence no
#       smartsim related items or non-standand library
#       items should be imported here.

# TODO: pkg_resources has been deprecated by PyPA. Currently we use it for its
#       packaging implementation, as we cannot assume a user will have `packaging`
#       prior to `pip install` time. We really only use pkg_resources for their
#       vendored version of `packaging.version.Version` so we should probably try
#       to remove
# https://setuptools.pypa.io/en/latest/pkg_resources.html

import pkg_resources
from pkg_resources import packaging  # type: ignore

Version = packaging.version.Version
InvalidVersion = packaging.version.InvalidVersion
DbEngine = t.Literal["REDIS", "KEYDB"]


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


class VersionConflictError(SetupError):
    """An error for when version numbers of some library/package/program/etc
    do not match and build may not be able to continue
    """

    def __init__(
        self,
        name: str,
        current_version: "Version_",
        target_version: "Version_",
        msg: t.Optional[str] = None,
    ) -> None:
        if msg is None:
            msg = (
                f"Incompatible version for {name} detected: "
                f"{name} {target_version} requested but {name} {current_version} "
                "installed."
            )
        super().__init__(msg)
        self.name = name
        self.current_version = current_version
        self.target_version = target_version


# so as to not conflict with pkg_resources.packaging.version.Version
# pylint: disable-next=invalid-name
class Version_(str):
    """A subclass of pkg_resources.packaging.version.Version that
    includes some helper methods for comparing versions.
    """

    @staticmethod
    def _convert_to_version(
        vers: t.Union[
            str, Iterable[packaging.version.Version], packaging.version.Version
        ],
    ) -> t.Any:
        if isinstance(vers, Version):
            return vers
        if isinstance(vers, str):
            return Version(vers)
        if isinstance(vers, Iterable):
            return Version(".".join((str(item) for item in vers)))

        raise InvalidVersion(vers)

    @property
    def major(self) -> int:
        # Version(self).major doesn't work for all Python distributions
        # see https://github.com/lebedov/python-pdfbox/issues/28
        return int(pkg_resources.parse_version(self).base_version.split(".")[0])

    @property
    def minor(self) -> int:
        return int(pkg_resources.parse_version(self).base_version.split(".")[1])

    @property
    def micro(self) -> int:
        return int(pkg_resources.parse_version(self).base_version.split(".")[2])

    @property
    def patch(self) -> str:
        # return micro with string modifier i.e. 1.2.3+cpu -> 3+cpu
        return str(pkg_resources.parse_version(self)).split(".")[2]

    def __gt__(self, cmp: t.Any) -> bool:
        try:
            return bool(Version(self).__gt__(self._convert_to_version(cmp)))
        except InvalidVersion:
            return super().__gt__(cmp)

    def __lt__(self, cmp: t.Any) -> bool:
        try:
            return bool(Version(self).__lt__(self._convert_to_version(cmp)))
        except InvalidVersion:
            return super().__lt__(cmp)

    def __eq__(self, cmp: t.Any) -> bool:
        try:
            return bool(Version(self).__eq__(self._convert_to_version(cmp)))
        except InvalidVersion:
            return super().__eq__(cmp)

    def __ge__(self, cmp: t.Any) -> bool:
        try:
            return bool(Version(self).__ge__(self._convert_to_version(cmp)))
        except InvalidVersion:
            return super().__ge__(cmp)

    def __le__(self, cmp: t.Any) -> bool:
        try:
            return bool(Version(self).__le__(self._convert_to_version(cmp)))
        except InvalidVersion:
            return super().__le__(cmp)

    def __hash__(self) -> int:
        return hash(Version(self))


def get_env(var: str, default: str) -> str:
    return os.environ.get(var, default)


class RedisAIVersion(Version_):
    """A subclass of Version_ that holds the dependency sets for RedisAI

    this class serves two purposes:

    1. It is used to populate the [ml] ``extras_require`` of the setup.py.
    This is because the RedisAI version will determine which ML based
    dependencies are required.

    2. Used to set the default values for PyTorch, TF, and ONNX
    given the SMARTSIM_REDISAI env var set by the user.

    NOTE: Torch requires additional information depending on whether
    CPU or GPU support is requested
    """

    defaults = {
        "1.2.5": {
            "tensorflow": "2.6.2",
            "onnx": "1.9.0",
            "skl2onnx": "1.10.3",
            "onnxmltools": "1.10.0",
            "scikit-learn": "1.0.2",
            "torch": "1.9.1",
            "torch_cpu_suffix": "+cpu",
            "torch_cuda_suffix": "+cu111",
            "torchvision": "0.10.1",
        },
        "1.2.7": {
            "tensorflow": "2.8.0",
            "onnx": "1.11.0",
            "skl2onnx": "1.11.1",
            "onnxmltools": "1.11.1",
            "scikit-learn": "1.1.1",
            "torch": "1.11.0",
            "torch_cpu_suffix": "+cpu",
            "torch_cuda_suffix": "+cu113",
            "torchvision": "0.12.0",
        },
    }
    # Remove options with unsported wheels for python>=3.10
    if sys.version_info >= (3, 10):
        defaults.pop("1.2.5")
        defaults["1.2.7"].pop("onnx")
        defaults["1.2.7"].pop("skl2onnx")
        defaults["1.2.7"].pop("onnxmltools")
        defaults["1.2.7"].pop("scikit-learn")
    # Remove incompatible RAI versions for OSX
    if sys.platform == "darwin":
        defaults.pop("1.2.5", None)

    def __init__(self, vers: str) -> None:  # pylint: disable=super-init-not-called
        min_rai_version = min(Version_(ver) for ver in self.defaults)
        if min_rai_version > vers:
            raise SetupError(
                f"RedisAI version must be greater than or equal to {min_rai_version}"
            )
        if vers not in self.defaults:
            if vers.startswith("1.2"):
                # resolve to latest version for 1.2.x
                # the str representation will still be 1.2.x
                self.version = "1.2.7"
            else:
                raise SetupError(
                    (
                        f"Invalid RedisAI version {vers}. Options are "
                        f"{self.defaults.keys()}"
                    )
                )
        else:
            self.version = vers

    def __getattr__(self, name: str) -> str:
        try:
            return self.defaults[self.version][name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'\n\n"
                "This is likely a problem with the SmartSim build process;"
                "if this problem persists please log a new issue at "
                "https://github.com/CrayLabs/SmartSim/issues "
                "or get in contact with us at "
                "https://www.craylabs.org/docs/community.html"
            ) from None

    def get_defaults(self) -> t.Dict[str, str]:
        return self.defaults[self.version].copy()


class Versioner:
    """Versioner is responsible for managing all the versions
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

    Default versions for SmartSim, SmartRedis, Redis, and RedisAI are
    all set here. Setting a default version for RedisAI also dictates
    default versions of the machine learning libraries.
    """

    # compatible Python version
    PYTHON_MIN = Version_("3.8.0")

    # Versions
    SMARTSIM = Version_(get_env("SMARTSIM_VERSION", "0.5.1"))
    SMARTREDIS = Version_(get_env("SMARTREDIS_VERSION", "0.4.2"))
    SMARTSIM_SUFFIX = get_env("SMARTSIM_SUFFIX", "")

    # Redis
    REDIS = Version_(get_env("SMARTSIM_REDIS", "7.0.5"))
    REDIS_URL = get_env("SMARTSIM_REDIS_URL", "https://github.com/redis/redis.git/")
    REDIS_BRANCH = get_env("SMARTSIM_REDIS_BRANCH", REDIS)

    # RedisAI
    REDISAI = RedisAIVersion(get_env("SMARTSIM_REDISAI", "1.2.7"))
    REDISAI_URL = get_env(
        "SMARTSIM_REDISAI_URL", "https://github.com/RedisAI/RedisAI.git/"
    )
    REDISAI_BRANCH = get_env("SMARTSIM_REDISAI_BRANCH", f"v{REDISAI}")

    # ML/DL (based on RedisAI version defaults)
    # torch can be set by the user because we download that for them
    TORCH = Version_(get_env("SMARTSIM_TORCH", REDISAI.torch))
    TORCHVISION = Version_(get_env("SMARTSIM_TORCHVIS", REDISAI.torchvision))
    TORCH_CPU_SUFFIX = Version_(get_env("TORCH_CPU_SUFFIX", REDISAI.torch_cpu_suffix))
    TORCH_CUDA_SUFFIX = Version_(
        get_env("TORCH_CUDA_SUFFIX", REDISAI.torch_cuda_suffix)
    )

    # TensorFlow and ONNX only use the defaults, but these are not built into
    # the RedisAI package and therefore the user is free to pick other versions.
    TENSORFLOW = Version_(REDISAI.tensorflow)
    try:
        ONNX = Version_(REDISAI.onnx)
    except AttributeError:
        ONNX = None

    def as_dict(self, db_name: DbEngine = "REDIS") -> t.Dict[str, t.Any]:
        packages = [
            "SMARTSIM",
            "SMARTREDIS",
            db_name,
            "REDISAI",
            "TORCH",
            "TENSORFLOW",
        ]
        versions = [
            self.SMARTSIM,
            self.SMARTREDIS,
            self.REDIS,
            self.REDISAI,
            self.TORCH,
            self.TENSORFLOW,
        ]
        if self.ONNX:
            packages.append("ONNX")
            versions.append(self.ONNX)
        vers = {"Packages": packages, "Versions": versions}
        return vers

    def ml_extras_required(self) -> t.Dict[str, t.List[str]]:
        """Optional ML/DL dependencies we suggest for the user.

        The defaults are based on the RedisAI version
        """
        ml_defaults = self.REDISAI.get_defaults()

        # remove torch-related fields as they are subject to change
        # by having the user change hardware (cpu/gpu)
        _torch_fields = [
            "torch",
            "torchvision",
            "torch_cpu_suffix",
            "torch_cuda_suffix",
        ]
        for field in _torch_fields:
            ml_defaults.pop(field)

        return {
            "ml": [f"{lib}=={vers}" for lib, vers in ml_defaults.items()]
        }

    @staticmethod
    def get_sha(setup_py_dir: Path) -> str:
        """Get the git sha of the current branch"""
        try:
            rev_cmd = ["git", "rev-parse", "HEAD"]
            git_rev = subprocess.check_output(rev_cmd, cwd=setup_py_dir.absolute())
            sha = git_rev.decode("ascii").strip()

            return sha[:7]
        except Exception:
            # return empty string if not in git-repo
            return ""

    def write_version(self, setup_py_dir: Path) -> str:
        """
        Write version info to version.py

        Use git_sha in the case where smartsim suffix is set in the environment
        """
        version = str(self.SMARTSIM)

        if self.SMARTSIM_SUFFIX:
            version += f"+{self.SMARTSIM_SUFFIX}"

            # wheel build (python -m build) won't include git sha
            if git_sha := self.get_sha(setup_py_dir):
                version += f".{git_sha}"

        version_file_path = setup_py_dir / "smartsim" / "version.py"
        with open(version_file_path, "w", encoding="utf-8") as version_file:
            version_file.write("# This file is automatically generated by setup.py\n")
            version_file.write("# do not edit this file manually.\n\n")

            version_file.write(f"__version__ = '{version}'\n")
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
    JOBS = int(os.environ.get("BUILD_JOBS", 1))

    # check for CC/GCC/ETC
    CHECKS = int(os.environ.get("NO_CHECKS", 0))
    PLATFORM = sys.platform

    def __init__(self, checks: bool = True) -> None:
        if checks:
            self.check_dependencies()

    def check_dependencies(self) -> None:
        deps = ["git", "git-lfs", "make", "wget", "cmake", self.CC, self.CXX]
        if int(self.CHECKS) == 0:
            for dep in deps:
                self.check_build_dependency(dep)

    def __call__(self) -> t.Dict[str, str]:
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

    def as_dict(self) -> t.Dict[str, t.List[str]]:
        variables: t.List[str] = [
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "MALLOC",
            "JOBS",
            "PYTHON_VERSION",
            "PLATFORM",
        ]
        values: t.List[str] = [
            self.CC,
            self.CXX,
            self.CFLAGS,
            self.CXXFLAGS,
            self.MALLOC,
            str(self.JOBS),
            self.python_version,
            self.PLATFORM,
        ]
        env = {"Environment": variables, "Values": values}
        return env

    @property
    def python_version(self) -> str:
        return platform.python_version()

    @staticmethod
    def is_compatible_python(python_min: float) -> bool:
        """Detect if system Python is too old"""
        sys_py = sys.version_info
        system_python = Version_(f"{sys_py.major}.{sys_py.minor}.{sys_py.micro}")
        return system_python > python_min

    @classmethod
    def is_windows(cls) -> bool:
        return cls.PLATFORM in ["win32", "cygwin", "msys"]

    @classmethod
    def is_macos(cls) -> bool:
        return cls.PLATFORM == "darwin"

    @staticmethod
    def get_cudnn_env() -> t.Optional[t.Dict[str, str]]:
        """Collect the environment variables needed for Caffe (Pytorch)
        and throw an error if they are not found

        Specifically make sure to set at least one set of:
            - CUDNN_LIBRARY and CUDNN_INCLUDE_DIR
                or
            - CUDNN_LIBRARY_PATH and CUDNN_INCLUDE_PATH
        """
        env = {
            "CUDNN_LIBRARY": os.environ.get("CUDNN_LIBRARY", "env-var-not-found"),
            "CUDNN_INCLUDE_DIR": os.environ.get(
                "CUDNN_INCLUDE_DIR", "env-var-not-found"
            ),
            "CUDNN_LIBRARY_PATH": os.environ.get(
                "CUDNN_LIBRARY_PATH", "env-var-not-found"
            ),
            "CUDNN_INCLUDE_PATH": os.environ.get(
                "CUDNN_INCLUDE_PATH", "env-var-not-found"
            ),
        }
        torch_cudnn_vars = ["CUDNN_LIBRARY", "CUDNN_INCLUDE_DIR"]
        caffe_cudnn_vars = ["CUDNN_INCLUDE_PATH", "CUDNN_LIBRARY_PATH"]

        torch_set = all(var in os.environ for var in torch_cudnn_vars)
        caffe_set = all(var in os.environ for var in caffe_cudnn_vars)

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

    @staticmethod
    def check_build_dependency(command: str) -> None:
        # TODO expand this to parse and check versions.
        try:
            subprocess.check_call(
                [command, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            raise SetupError(f"{command} must be installed to build SmartSim") from None

    @classmethod
    def check_installed(
        cls, package: str, version: t.Optional[Version_] = None
    ) -> bool:
        """Check if a package is installed. If version is provided, check if
        it's a compatible version. (major and minor the same)
        """
        try:
            installed = cls.get_py_package_version(package)
        except importlib.metadata.PackageNotFoundError:
            return False
        if version:
            # detect if major or minor versions differ
            if installed.major != version.major or installed.minor != version.minor:
                raise VersionConflictError(package, installed, version)
        return True

    @staticmethod
    def get_py_package_version(package: str) -> Version_:
        return Version_(importlib.metadata.version(package))
