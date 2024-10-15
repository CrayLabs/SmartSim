# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

from packaging.version import InvalidVersion, Version, parse

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


# so as to not conflict with pkg_resources.packaging.version.Version
# pylint: disable-next=invalid-name
class Version_(str):
    """A subclass of pkg_resources.packaging.version.Version that
    includes some helper methods for comparing versions.
    """

    @staticmethod
    def _convert_to_version(
        vers: t.Union[str, Iterable[Version], Version],
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
        return int(parse(self).base_version.split(".", maxsplit=1)[0])

    @property
    def minor(self) -> int:
        return int(parse(self).base_version.split(".", maxsplit=2)[1])

    @property
    def micro(self) -> int:
        return int(parse(self).base_version.split(".", maxsplit=3)[2])

    @property
    def patch(self) -> str:
        # return micro with string modifier i.e. 1.2.3+cpu -> 3+cpu
        return str(parse(self)).split(".")[2]

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

    Default versions for SmartSim, Redis, and RedisAI are specified here.
    """

    # compatible Python version
    PYTHON_MIN = Version_("3.9.0")

    # Versions
    SMARTSIM = Version_(get_env("SMARTSIM_VERSION", "0.8.0"))
    SMARTSIM_SUFFIX = get_env("SMARTSIM_SUFFIX", "")

    # Redis
    REDIS = Version_(get_env("SMARTSIM_REDIS", "7.2.4"))
    REDIS_URL = get_env("SMARTSIM_REDIS_URL", "https://github.com/redis/redis.git")
    REDIS_BRANCH = get_env("SMARTSIM_REDIS_BRANCH", REDIS)

    # RedisAI
    REDISAI = "1.2.7"
    REDISAI_URL = get_env(
        "SMARTSIM_REDISAI_URL", "https://github.com/RedisAI/RedisAI.git"
    )
    REDISAI_BRANCH = get_env("SMARTSIM_REDISAI_BRANCH", f"v{REDISAI}")

    def as_dict(self, db_name: DbEngine = "REDIS") -> t.Dict[str, t.Tuple[str, ...]]:
        pkg_map = {
            "SMARTSIM": self.SMARTSIM,
            db_name: self.REDIS,
            "REDISAI": self.REDISAI,
        }
        return {"Packages": tuple(pkg_map), "Versions": tuple(pkg_map.values())}

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
        deps = ["git", "make", "wget", "cmake", self.CC, self.CXX]
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

    @staticmethod
    def get_py_package_version(package: str) -> Version_:
        return Version_(importlib.metadata.version(package))
