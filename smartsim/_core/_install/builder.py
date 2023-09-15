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
import re
import shutil
import stat
import subprocess
import sys
import typing as t
from pathlib import Path
from shutil import which
from subprocess import SubprocessError

# NOTE: This will be imported by setup.py and hence no
#       smartsim related items should be imported into
#       this file.

# TODO:
#   - check cmake version and use system if possible to avoid conflicts

TRedisAIBackendStr = t.Literal["tensorflow", "torch", "onnxruntime", "tflite"]


def expand_exe_path(exe: str) -> str:
    """Takes an executable and returns the full path to that executable

    :param exe: executable or file
    :type exe: str
    :raises TypeError: if file is not an executable
    :raises FileNotFoundError: if executable cannot be found
    """

    # which returns none if not found
    in_path = which(exe)
    if not in_path:
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return os.path.abspath(exe)
        if os.path.isfile(exe) and not os.access(exe, os.X_OK):
            raise TypeError(f"File, {exe}, is not an executable")
        raise FileNotFoundError(f"Could not locate executable {exe}")
    return os.path.abspath(in_path)


class BuildError(Exception):
    pass


class Builder:
    """Base class for building third-party libraries"""

    url_regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # pylint: disable=line-too-long
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    def __init__(
        self, env: t.Dict[str, t.Any], jobs: t.Optional[int] = 1, verbose: bool = False
    ) -> None:
        # build environment from buildenv
        self.env = env

        # Find _core directory and set up paths
        _core_dir = Path(os.path.abspath(__file__)).parent.parent

        dependency_path = _core_dir
        if os.getenv("SMARTSIM_DEP_PATH"):
            dependency_path = Path(os.environ["SMARTSIM_DEP_PATH"])

        self.build_dir = _core_dir / ".third-party"

        self.bin_path = dependency_path / "bin"
        self.lib_path = dependency_path / "lib"

        # Set wether build process will output to std output
        self.out: t.Optional[int] = subprocess.DEVNULL
        self.verbose = verbose
        if self.verbose:
            self.out = None

        # make build directory "SmartSim/smartsim/_core/.third-party"
        if not self.build_dir.is_dir():
            self.build_dir.mkdir()
        if dependency_path == _core_dir:
            if not self.bin_path.is_dir():
                self.bin_path.mkdir()
            if not self.lib_path.is_dir():
                self.lib_path.mkdir()

        self.jobs = jobs

    # implemented in base classes
    @property
    def is_built(self) -> bool:
        raise NotImplementedError

    def build_from_git(self, git_url: str, branch: str, device: str = "cpu") -> None:
        raise NotImplementedError

    @staticmethod
    def binary_path(binary: str) -> str:
        binary_ = shutil.which(binary)
        if binary_:
            return binary_
        raise BuildError(f"{binary} not found in PATH")

    @staticmethod
    def copy_file(
        src: t.Union[str, Path], dst: t.Union[str, Path], set_exe: bool = False
    ) -> None:
        shutil.copyfile(src, dst)
        if set_exe:
            Path(dst).chmod(stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)

    def copy_dir(
        self, src: t.Union[str, Path], dst: t.Union[str, Path], set_exe: bool = False
    ) -> None:
        src = Path(src)
        dst = Path(dst)
        dst.mkdir(exist_ok=True)
        # copy directory contents
        for content in src.glob("*"):
            if content.is_dir():
                self.copy_dir(content, dst / content.name, set_exe=set_exe)
            else:
                self.copy_file(content, dst / content.name, set_exe=set_exe)

    def is_valid_url(self, url: str) -> bool:
        return re.match(self.url_regex, url) is not None

    def cleanup(self) -> None:
        if self.build_dir.is_dir():
            shutil.rmtree(str(self.build_dir))

    def run_command(
        self,
        cmd: t.List[str],
        shell: bool = False,
        out: t.Optional[int] = None,
        cwd: t.Union[str, Path, None] = None,
    ) -> None:
        # option to manually disable output if necessary
        if not out:
            out = self.out
        try:
            # pylint: disable-next=consider-using-with
            proc = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=out,
                cwd=cwd,
                shell=shell,
                env=self.env,
            )
            error = proc.communicate()[1].decode("utf-8")
            if proc.returncode != 0:
                raise BuildError(error)
        except (OSError, SubprocessError) as e:
            raise BuildError(e) from e


class DatabaseBuilder(Builder):
    """Class to build Redis or KeyDB from Source
    Supported build methods:
     - from git
    See buildenv.py for buildtime configuration of Redis/KeyDB
    version and url.
    """

    def __init__(
        self,
        build_env: t.Optional[t.Dict[str, t.Any]] = None,
        malloc: str = "libc",
        jobs: t.Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(build_env or {}, jobs=jobs, verbose=verbose)
        self.malloc = malloc

    @property
    def is_built(self) -> bool:
        """Check if Redis or KeyDB is built"""
        bin_files = {file.name for file in self.bin_path.iterdir()}
        redis_files = {"redis-server", "redis-cli"}
        keydb_files = {"keydb-server", "keydb-cli"}
        return redis_files.issubset(bin_files) or keydb_files.issubset(bin_files)

    def build_from_git(self, git_url: str, branch: str, device: str = "cpu") -> None:
        """Build Redis from git
        :param git_url: url from which to retrieve Redis
        :type git_url: str
        :param branch: branch to checkout
        :type branch: str
        """
        # pylint: disable=too-many-locals
        database_name = "keydb" if "KeyDB" in git_url else "redis"
        database_build_path = Path(self.build_dir, database_name.lower())

        # remove git directory if it exists as it should
        # really never exist as we delete after build
        redis_build_path = Path(self.build_dir, "redis")
        keydb_build_path = Path(self.build_dir, "keydb")
        if redis_build_path.is_dir():
            shutil.rmtree(str(redis_build_path))
        if keydb_build_path.is_dir():
            shutil.rmtree(str(keydb_build_path))

        # Check database URL
        if not self.is_valid_url(git_url):
            raise BuildError(f"Malformed {database_name} URL: {git_url}")

        # clone Redis
        clone_cmd = [
            self.binary_path("git"),
            "clone",
            git_url,
            "--branch",
            branch,
            "--depth",
            "1",
            database_name,
        ]
        self.run_command(clone_cmd, cwd=self.build_dir)

        # build Redis
        build_cmd = [
            self.binary_path("make"),
            "-j",
            str(self.jobs),
            f"MALLOC={self.malloc}",
        ]
        self.run_command(build_cmd, cwd=str(database_build_path))

        # move redis binaries to smartsim/smartsim/_core/bin
        database_src_dir = database_build_path / "src"
        server_source = database_src_dir / (database_name.lower() + "-server")
        server_destination = self.bin_path / (database_name.lower() + "-server")
        cli_source = database_src_dir / (database_name.lower() + "-cli")
        cli_destination = self.bin_path / (database_name.lower() + "-cli")
        self.copy_file(server_source, server_destination, set_exe=True)
        self.copy_file(cli_source, cli_destination, set_exe=True)

        # validate install -- redis-server
        core_path = Path(os.path.abspath(__file__)).parent.parent
        dependency_path = os.environ.get("SMARTSIM_DEP_INSTALL_PATH", core_path)
        bin_path = Path(dependency_path, "bin").resolve()
        try:
            database_exe = next(bin_path.glob("*-server"))
            database = Path(os.environ.get("REDIS_PATH", database_exe)).resolve()
            _ = expand_exe_path(str(database))
        except (TypeError, FileNotFoundError) as e:
            raise BuildError("Installation of redis-server failed!") from e

        # validate install -- redis-cli
        try:
            redis_cli_exe = next(bin_path.glob("*-cli"))
            redis_cli = Path(os.environ.get("REDIS_CLI_PATH", redis_cli_exe)).resolve()
            _ = expand_exe_path(str(redis_cli))
        except (TypeError, FileNotFoundError) as e:
            raise BuildError("Installation of redis-cli failed!") from e


class RedisAIBuilder(Builder):
    """Class to build RedisAI from Source
    Supported build method:
     - from git
    See buildenv.py for buildtime configuration of RedisAI
    version and url.
    """

    def __init__(
        self,
        build_env: t.Optional[t.Dict[str, t.Any]] = None,
        torch_dir: str = "",
        libtf_dir: str = "",
        build_torch: bool = True,
        build_tf: bool = True,
        build_onnx: bool = False,
        jobs: t.Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(build_env or {}, jobs=jobs, verbose=verbose)
        self.rai_install_path: t.Optional[Path] = None

        # convert to int for RAI build script
        self._torch = build_torch
        self._tf = build_tf
        self._onnx = build_onnx
        self.libtf_dir = libtf_dir
        self.torch_dir = torch_dir

    @property
    def rai_build_path(self) -> Path:
        return Path(self.build_dir, "RedisAI")

    @property
    def is_built(self) -> bool:
        server = self.lib_path.joinpath("backends").is_dir()
        cli = self.lib_path.joinpath("redisai.so").is_file()
        return server and cli

    @property
    def build_torch(self) -> bool:
        return self._torch

    @property
    def fetch_torch(self) -> bool:
        return self.build_torch and not self.torch_dir

    @property
    def build_tf(self) -> bool:
        return self._tf

    @property
    def fetch_tf(self) -> bool:
        return self.build_tf and not self.libtf_dir

    @property
    def build_onnx(self) -> bool:
        return self._onnx

    @property
    def fetch_onnx(self) -> bool:
        return self.build_onnx

    def copy_tf_cmake(self) -> None:
        """Copy the FindTensorFlow.cmake file to the build directory
        as the version included in RedisAI is out of date for us.
        Note: opt/cmake/modules removed in RedisAI v1.2.5
        """
        # remove the previous version
        tf_cmake = self.rai_build_path / "opt/cmake/modules/FindTensorFlow.cmake"
        tf_cmake.resolve()
        if tf_cmake.is_file():
            tf_cmake.unlink()
            # copy ours in
            self.copy_file(
                self.bin_path / "modules/FindTensorFlow.cmake", tf_cmake, set_exe=False
            )

    def symlink_libtf(self, device: str) -> None:
        """Add symbolic link to available libtensorflow in RedisAI deps.

        :param device: cpu or gpu
        :type device: str
        """
        rai_deps_path = sorted(
            self.rai_build_path.glob(os.path.join("deps", f"*{device}*"))
        )
        if not rai_deps_path:
            raise FileNotFoundError("Could not find RedisAI 'deps' directory")

        # There should only be one path for a given device,
        # and this should hold even if in the future we use
        # an external build of RedisAI
        rai_libtf_path = rai_deps_path[0] / "libtensorflow"
        rai_libtf_path.resolve()
        if rai_libtf_path.is_dir():
            shutil.rmtree(rai_libtf_path)

        os.makedirs(rai_libtf_path)
        libtf_path = Path(self.libtf_dir).resolve()

        # Copy include directory to deps/libtensorflow
        include_src_path = libtf_path / "include"
        if not include_src_path.exists():
            raise FileNotFoundError(f"Could not find include directory in {libtf_path}")
        os.symlink(include_src_path, rai_libtf_path / "include")

        # RedisAI expects to find a lib directory, which is only
        # available in some distributions.
        rai_libtf_lib_dir = rai_libtf_path / "lib"
        os.makedirs(rai_libtf_lib_dir)
        src_libtf_lib_dir = libtf_path / "lib"
        # If the lib directory existed in the libtensorflow distribution,
        # copy its content, otherwise gather library files from
        # libtensorflow base dir and copy them into destination lib dir
        if src_libtf_lib_dir.is_dir():
            library_files = sorted(src_libtf_lib_dir.glob("*"))
            if not library_files:
                raise FileNotFoundError(
                    f"Could not find libtensorflow library files in {src_libtf_lib_dir}"
                )
        else:
            library_files = sorted(libtf_path.glob("lib*.so*"))
            if not library_files:
                raise FileNotFoundError(
                    f"Could not find libtensorflow library files in {libtf_path}"
                )

        for src_file in library_files:
            dst_file = rai_libtf_lib_dir / src_file.name
            if not dst_file.is_file():
                os.symlink(src_file, dst_file)

    def build_from_git(self, git_url: str, branch: str, device: str = "cpu") -> None:
        """Build RedisAI from git

        :param git_url: url from which to retrieve RedisAI
        :type git_url: str
        :param branch: branch to checkout
        :type branch: str
        :param device: cpu or gpu
        :type device: str
        """
        # delete previous build dir (should never be there)
        if self.rai_build_path.is_dir():
            shutil.rmtree(self.rai_build_path)

        # Check RedisAI URL
        if not self.is_valid_url(git_url):
            raise BuildError(f"Malformed RedisAI URL: {git_url}")

        # clone RedisAI
        clone_cmd = [
            self.binary_path("env"),
            "GIT_LFS_SKIP_SMUDGE=1",
            "git",
            "clone",
            "--recursive",
            git_url,
        ]

        checkout_osx_fix: t.List[str] = []

        # Circumvent a bad `get_deps.sh` script from RAI on 1.2.7 with ONNX
        # TODO: Look for a better way to do this or wait for RAI patch
        if sys.platform == "darwin" and branch == "v1.2.7" and self.build_onnx:
            # Clone RAI patch commit for OSX
            clone_cmd += ["RedisAI"]
            checkout_osx_fix = [
                "git",
                "checkout",
                "634916c722e718cc6ea3fad46e63f7d798f9adc2",
            ]
        else:
            # Clone RAI release commit
            clone_cmd += [
                "--branch",
                branch,
                "--depth=1",
                "RedisAI",
            ]

        self.run_command(clone_cmd, out=subprocess.DEVNULL, cwd=self.build_dir)
        if checkout_osx_fix:
            self.run_command(
                checkout_osx_fix, out=subprocess.DEVNULL, cwd=self.rai_build_path
            )

        # copy FindTensorFlow.cmake to RAI cmake dir
        self.copy_tf_cmake()

        # get RedisAI dependencies
        dep_cmd = self._rai_build_env_prefix(
            with_pt=self.build_torch,
            with_tf=self.build_tf,
            with_ort=self.build_onnx,
            extra_env={"VERBOSE": "1"},
        )

        dep_cmd.extend(
            [
                self.binary_path("bash"),
                str(self.rai_build_path / "get_deps.sh"),
                str(device),
            ]
        )

        self.run_command(
            dep_cmd,
            out=subprocess.DEVNULL,  # suppress this as it's not useful
            cwd=self.rai_build_path,
        )

        if self.libtf_dir and device:
            self.symlink_libtf(device)

        build_cmd = self._rai_build_env_prefix(
            with_pt=self.build_torch,
            with_tf=self.build_tf,
            with_ort=self.build_onnx,
            extra_env={"GPU": "1" if device == "gpu" else "0"},
        )

        if self.torch_dir:
            self.env["Torch_DIR"] = str(self.torch_dir)

        build_cmd.extend(
            [
                self.binary_path("make"),
                "-C",
                str(self.rai_build_path / "opt"),
                "-j",
                f"{self.jobs}",
                "build",
            ]
        )
        self.run_command(build_cmd, cwd=self.rai_build_path)

        self._install_backends(device)
        if self.user_supplied_backend("torch"):
            self._move_torch_libs()
        self.cleanup()

    def user_supplied_backend(self, backend: TRedisAIBackendStr) -> bool:
        if backend == "torch":
            return bool(self.build_torch and not self.fetch_torch)
        if backend == "tensorflow":
            return bool(self.build_tf and not self.fetch_tf)
        if backend == "onnxruntime":
            return bool(self.build_onnx and not self.fetch_onnx)
        if backend == "tflite":
            return False
        raise BuildError(f"Unrecognized backend requested {backend}")

    def _rai_build_env_prefix(
        self,
        with_tf: bool,
        with_pt: bool,
        with_ort: bool,
        extra_env: t.Optional[t.Dict[str, str]] = None,
    ) -> t.List[str]:
        extra_env = extra_env or {}
        return [
            self.binary_path("env"),
            f"WITH_PT={1 if with_pt else 0}",
            f"WITH_TF={1 if with_tf else 0}",
            "WITH_TFLITE=0",  # never use TF Lite (for now)
            f"WITH_ORT={1 if with_ort else 0}",
            *(f"{key}={val}" for key, val in extra_env.items()),
        ]

    def _install_backends(self, device: str) -> None:
        """Move backend libraries to smartsim/_core/lib/
        :param device: cpu or cpu
        :type device: str
        """
        self.rai_install_path = self.rai_build_path.joinpath(
            f"install-{device}"
        ).resolve()
        rai_lib = self.rai_install_path / "redisai.so"
        rai_backends = self.rai_install_path / "backends"

        if rai_lib.is_file() and rai_backends.is_dir():
            self.copy_dir(rai_backends, self.lib_path / "backends", set_exe=True)
            self.copy_file(rai_lib, self.lib_path / "redisai.so", set_exe=True)

    def _move_torch_libs(self) -> None:
        """Move pip install torch libraries
        Since we use pip installed torch libraries for building
        RedisAI, we need to move them into the LD_runpath of redisai.so
        in the smartsim/_core/lib directory.
        """
        ss_rai_torch_path = self.lib_path / "backends" / "redisai_torch"
        ss_rai_torch_lib_path = ss_rai_torch_path / "lib"

        # retrieve torch shared libraries and copy to the
        # smartsim/_core/lib/backends/redisai_torch/lib dir
        # self.torch_dir should be /path/to/torch/share/cmake/Torch
        # so we take the great grandparent here
        pip_torch_path = Path(self.torch_dir).parent.parent.parent
        pip_torch_lib_path = pip_torch_path / "lib"

        self.copy_dir(pip_torch_lib_path, ss_rai_torch_lib_path, set_exe=True)

        # also move the openmp files if on a mac
        if sys.platform == "darwin":
            dylibs = pip_torch_path / ".dylibs"
            self.copy_dir(dylibs, ss_rai_torch_path / ".dylibs", set_exe=True)
