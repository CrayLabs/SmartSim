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

# pylint: disable=too-many-lines

import concurrent.futures
import enum
import fileinput
import itertools
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import typing as t
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import SubprocessError

# NOTE: This will be imported by setup.py and hence no smartsim related
# items should be imported into this file.

# TODO: check cmake version and use system if possible to avoid conflicts

TRedisAIBackendStr = t.Literal["tensorflow", "torch", "onnxruntime", "tflite"]
_PathLike = t.Union[str, "os.PathLike[str]"]
_T = t.TypeVar("_T")
_U = t.TypeVar("_U")


def expand_exe_path(exe: str) -> str:
    """Takes an executable and returns the full path to that executable

    :param exe: executable or file
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


class Architecture(enum.Enum):
    X64 = ("x86_64", "amd64")
    ARM64 = ("arm64",)

    @classmethod
    def from_str(cls, string: str, /) -> "Architecture":
        string = string.lower()
        for type_ in cls:
            if string in type_.value:
                return type_
        raise BuildError(f"Unrecognized or unsupported architecture: {string}")


class Device(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class OperatingSystem(enum.Enum):
    LINUX = ("linux", "linux2")
    DARWIN = ("darwin",)

    @classmethod
    def from_str(cls, string: str, /) -> "OperatingSystem":
        string = string.lower()
        for type_ in cls:
            if string in type_.value:
                return type_
        raise BuildError(f"Unrecognized or unsupported operating system: {string}")


class Platform(t.NamedTuple):
    os: OperatingSystem
    architecture: Architecture


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
        self,
        env: t.Dict[str, str],
        jobs: int = 1,
        _os: OperatingSystem = OperatingSystem.from_str(platform.system()),
        architecture: Architecture = Architecture.from_str(platform.machine()),
        verbose: bool = False,
    ) -> None:
        # build environment from buildenv
        self.env = env
        self._platform = Platform(_os, architecture)

        # Find _core directory and set up paths
        _core_dir = Path(os.path.abspath(__file__)).parent.parent

        dependency_path = _core_dir
        if os.getenv("SMARTSIM_DEP_PATH"):
            dependency_path = Path(os.environ["SMARTSIM_DEP_PATH"])

        self.build_dir = _core_dir / ".third-party"

        self.bin_path = dependency_path / "bin"
        self.lib_path = dependency_path / "lib"
        self.verbose = verbose

        # make build directory "SmartSim/smartsim/_core/.third-party"
        if not self.build_dir.is_dir():
            self.build_dir.mkdir()
        if dependency_path == _core_dir:
            if not self.bin_path.is_dir():
                self.bin_path.mkdir()
            if not self.lib_path.is_dir():
                self.lib_path.mkdir()

        self.jobs = jobs

    @property
    def out(self) -> t.Optional[int]:
        return None if self.verbose else subprocess.DEVNULL

    # implemented in base classes
    @property
    def is_built(self) -> bool:
        raise NotImplementedError

    def build_from_git(
        self, git_url: str, branch: str, device: Device = Device.CPU
    ) -> None:
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
        build_env: t.Optional[t.Dict[str, str]] = None,
        malloc: str = "libc",
        jobs: int = 1,
        _os: OperatingSystem = OperatingSystem.from_str(platform.system()),
        architecture: Architecture = Architecture.from_str(platform.machine()),
        verbose: bool = False,
    ) -> None:
        super().__init__(
            build_env or {},
            jobs=jobs,
            _os=_os,
            architecture=architecture,
            verbose=verbose,
        )
        self.malloc = malloc

    @property
    def is_built(self) -> bool:
        """Check if Redis or KeyDB is built"""
        bin_files = {file.name for file in self.bin_path.iterdir()}
        redis_files = {"redis-server", "redis-cli"}
        keydb_files = {"keydb-server", "keydb-cli"}
        return redis_files.issubset(bin_files) or keydb_files.issubset(bin_files)

    def build_from_git(
        self, git_url: str, branch: str, device: Device = Device.CPU
    ) -> None:
        """Build Redis from git
        :param git_url: url from which to retrieve Redis
        :param branch: branch to checkout
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

        clone_cmd = config_git_command(
            self._platform,
            [
                self.binary_path("git"),
                "clone",
                git_url,
                "--branch",
                branch,
                "--depth",
                "1",
                database_name,
            ],
        )

        # clone Redis
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


class _RAIBuildDependency(ABC):
    """An interface with a collection of magic methods so that
    ``RedisAIBuilder`` can fetch and place its own dependencies
    """

    @property
    @abstractmethod
    def __rai_dependency_name__(self) -> str: ...

    @abstractmethod
    def __place_for_rai__(self, target: _PathLike) -> Path: ...

    @staticmethod
    @abstractmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]: ...


def _place_rai_dep_at(
    target: _PathLike, verbose: bool
) -> t.Callable[[_RAIBuildDependency], Path]:
    def _place(dep: _RAIBuildDependency) -> Path:
        if verbose:
            print(f"Placing: '{dep.__rai_dependency_name__}'")
        path = dep.__place_for_rai__(target)
        if verbose:
            print(f"Placed: '{dep.__rai_dependency_name__}' at '{path}'")
        return path

    return _place


class RedisAIBuilder(Builder):
    """Class to build RedisAI from Source
    Supported build method:
     - from git
    See buildenv.py for buildtime configuration of RedisAI
    version and url.
    """

    def __init__(
        self,
        _os: OperatingSystem = OperatingSystem.from_str(platform.system()),
        architecture: Architecture = Architecture.from_str(platform.machine()),
        build_env: t.Optional[t.Dict[str, str]] = None,
        torch_dir: str = "",
        libtf_dir: str = "",
        build_torch: bool = True,
        build_tf: bool = True,
        build_onnx: bool = False,
        jobs: int = 1,
        verbose: bool = False,
        torch_with_mkl: bool = True,
    ) -> None:
        super().__init__(
            build_env or {},
            jobs=jobs,
            _os=_os,
            architecture=architecture,
            verbose=verbose,
        )

        self.rai_install_path: t.Optional[Path] = None

        # convert to int for RAI build script
        self._torch = build_torch
        self._tf = build_tf
        self._onnx = build_onnx
        self.libtf_dir = libtf_dir
        self.torch_dir = torch_dir

        # extra configuration options
        self.torch_with_mkl = torch_with_mkl

        # Sanity checks
        self._validate_platform()

    def _validate_platform(self) -> None:
        unsupported = []
        if self._platform not in _DLPackRepository.supported_platforms():
            unsupported.append("DLPack")
        if self.fetch_tf and (self._platform not in _TFArchive.supported_platforms()):
            unsupported.append("Tensorflow")
        if self.fetch_onnx and (
            self._platform not in _ORTArchive.supported_platforms()
        ):
            unsupported.append("ONNX")
        if self.fetch_torch and (
            self._platform not in _PTArchive.supported_platforms()
        ):
            unsupported.append("PyTorch")
        if unsupported:
            raise BuildError(
                f"The {', '.join(unsupported)} backend(s) are not supported "
                f"on {self._platform.os} with {self._platform.architecture}"
            )

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

    def get_deps_dir_path_for(self, device: Device) -> Path:
        def fail_to_format(reason: str) -> BuildError:  # pragma: no cover
            return BuildError(f"Failed to format RedisAI dependency path: {reason}")

        _os, architecture = self._platform
        if _os == OperatingSystem.DARWIN:
            os_ = "macos"
        elif _os == OperatingSystem.LINUX:
            os_ = "linux"
        else:  # pragma: no cover
            raise fail_to_format(f"Unknown operating system: {_os}")
        if architecture == Architecture.X64:
            arch = "x64"
        elif architecture == Architecture.ARM64:
            arch = "arm64v8"
        else:  # pragma: no cover
            raise fail_to_format(f"Unknown architecture: {architecture}")
        return self.rai_build_path / f"deps/{os_}-{arch}-{device.value}"

    def _get_deps_to_fetch_for(
        self, device: Device
    ) -> t.Tuple[_RAIBuildDependency, ...]:
        os_, arch = self._platform
        # TODO: It would be nice if the backend version numbers were declared
        #       alongside the python package version numbers so that all of the
        #       dependency versions were declared in single location.
        #       Unfortunately importing into this module is non-trivial as it
        #       is used as script in the SmartSim `setup.py`.

        # DLPack is always required
        fetchable_deps: t.List[_RAIBuildDependency] = [_DLPackRepository("v0.5_RAI")]
        if self.fetch_torch:
            pt_dep = _choose_pt_variant(os_)(arch, device, "2.0.1", self.torch_with_mkl)
            fetchable_deps.append(pt_dep)
        if self.fetch_tf:
            fetchable_deps.append(_TFArchive(os_, arch, device, "2.13.1"))
        if self.fetch_onnx:
            fetchable_deps.append(_ORTArchive(os_, device, "1.16.3"))

        return tuple(fetchable_deps)

    def symlink_libtf(self, device: Device) -> None:
        """Add symbolic link to available libtensorflow in RedisAI deps.

        :param device: cpu or gpu
        """
        rai_deps_path = sorted(
            self.rai_build_path.glob(os.path.join("deps", f"*{device.value}*"))
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

    def build_from_git(
        self, git_url: str, branch: str, device: Device = Device.CPU
    ) -> None:
        """Build RedisAI from git

        :param git_url: url from which to retrieve RedisAI
        :param branch: branch to checkout
        :param device: cpu or gpu
        """
        # delete previous build dir (should never be there)
        if self.rai_build_path.is_dir():
            shutil.rmtree(self.rai_build_path)

        # Check RedisAI URL
        if not self.is_valid_url(git_url):
            raise BuildError(f"Malformed RedisAI URL: {git_url}")

        # clone RedisAI
        clone_cmd = config_git_command(
            self._platform,
            [
                self.binary_path("env"),
                "GIT_LFS_SKIP_SMUDGE=1",
                "git",
                "clone",
                "--recursive",
                git_url,
                "--branch",
                branch,
                "--depth=1",
                os.fspath(self.rai_build_path),
            ],
        )

        self.run_command(clone_cmd, out=subprocess.DEVNULL, cwd=self.build_dir)
        self._fetch_deps_for(device)

        if self.libtf_dir and device.value:
            self.symlink_libtf(device)

        build_cmd = self._rai_build_env_prefix(
            with_pt=self.build_torch,
            with_tf=self.build_tf,
            with_ort=self.build_onnx,
            extra_env={"GPU": "1" if device == Device.GPU else "0"},
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

    def _fetch_deps_for(self, device: Device) -> None:
        if not self.rai_build_path.is_dir():
            raise BuildError("RedisAI build directory not found")

        deps_dir = self.get_deps_dir_path_for(device)
        deps_dir.mkdir(parents=True, exist_ok=True)
        if any(deps_dir.iterdir()):
            raise BuildError("RAI build dependency directory is not empty")
        to_fetch = self._get_deps_to_fetch_for(device)
        placed_paths = _threaded_map(
            _place_rai_dep_at(deps_dir, self.verbose), to_fetch
        )
        unique_placed_paths = {os.fspath(path.resolve()) for path in placed_paths}
        if len(unique_placed_paths) != len(to_fetch):
            raise BuildError(
                f"Expected to place {len(to_fetch)} dependencies, but only "
                f"found {len(unique_placed_paths)}"
            )

    def _install_backends(self, device: Device) -> None:
        """Move backend libraries to smartsim/_core/lib/
        :param device: cpu or cpu
        """
        self.rai_install_path = self.rai_build_path.joinpath(
            f"install-{device.value}"
        ).resolve()
        rai_lib = self.rai_install_path / "redisai.so"
        rai_backends = self.rai_install_path / "backends"

        if rai_backends.is_dir():
            self.copy_dir(rai_backends, self.lib_path / "backends", set_exe=True)
        if rai_lib.is_file():
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


def _threaded_map(fn: t.Callable[[_T], _U], items: t.Iterable[_T]) -> t.Sequence[_U]:
    items = tuple(items)
    if not items:  # No items so no work to do
        return ()
    num_workers = min(len(items), (os.cpu_count() or 4) * 5)
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        return tuple(pool.map(fn, items))


class _WebLocation(ABC):
    @property
    @abstractmethod
    def url(self) -> str: ...


class _WebGitRepository(_WebLocation):
    def clone(
        self,
        target: _PathLike,
        depth: t.Optional[int] = None,
        branch: t.Optional[str] = None,
    ) -> None:
        depth_ = ("--depth", str(depth)) if depth is not None else ()
        branch_ = ("--branch", branch) if branch is not None else ()
        _git("clone", "-q", *depth_, *branch_, self.url, os.fspath(target))


@t.final
@dataclass(frozen=True)
class _DLPackRepository(_WebGitRepository, _RAIBuildDependency):
    version: str

    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        return (
            (OperatingSystem.LINUX, Architecture.X64),
            (OperatingSystem.DARWIN, Architecture.X64),
            (OperatingSystem.DARWIN, Architecture.ARM64),
        )

    @property
    def url(self) -> str:
        return "https://github.com/RedisAI/dlpack.git"

    @property
    def __rai_dependency_name__(self) -> str:
        return f"dlpack@{self.url}"

    def __place_for_rai__(self, target: _PathLike) -> Path:
        target = Path(target) / "dlpack"
        self.clone(target, branch=self.version, depth=1)
        if not target.is_dir():
            raise BuildError("Failed to place dlpack")
        return target


class _WebArchive(_WebLocation):
    @property
    def name(self) -> str:
        _, name = self.url.rsplit("/", 1)
        return name

    def download(self, target: _PathLike) -> Path:
        target = Path(target)
        if target.is_dir():
            target = target / self.name
        file, _ = urllib.request.urlretrieve(self.url, target)
        return Path(file).resolve()


class _ExtractableWebArchive(_WebArchive, ABC):
    @abstractmethod
    def _extract_download(self, download_path: Path, target: _PathLike) -> None: ...

    def extract(self, target: _PathLike) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            arch_path = self.download(tmp_dir)
            self._extract_download(arch_path, target)


class _WebTGZ(_ExtractableWebArchive):
    def _extract_download(self, download_path: Path, target: _PathLike) -> None:
        with tarfile.open(download_path, "r") as tgz_file:
            tgz_file.extractall(target)


class _WebZip(_ExtractableWebArchive):
    def _extract_download(self, download_path: Path, target: _PathLike) -> None:
        with zipfile.ZipFile(download_path, "r") as zip_file:
            zip_file.extractall(target)


class WebTGZ(_WebTGZ):
    def __init__(self, url: str) -> None:
        self._url = url

    @property
    def url(self) -> str:
        return self._url


@dataclass(frozen=True)
class _PTArchive(_WebZip, _RAIBuildDependency):
    architecture: Architecture
    device: Device
    version: str
    with_mkl: bool

    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        # TODO: This will need to be revisited if the inheritance tree gets deeper
        return tuple(
            itertools.chain.from_iterable(
                var.supported_platforms() for var in _PTArchive.__subclasses__()
            )
        )

    @property
    def __rai_dependency_name__(self) -> str:
        return f"libtorch@{self.url}"

    @staticmethod
    def _patch_out_mkl(libtorch_root: Path) -> None:
        _modify_source_files(
            libtorch_root / "share/cmake/Caffe2/public/mkl.cmake",
            r"find_package\(MKL QUIET\)",
            "# find_package(MKL QUIET)",
        )

    def extract(self, target: _PathLike) -> None:
        super().extract(target)
        if not self.with_mkl:
            self._patch_out_mkl(Path(target))

    def __place_for_rai__(self, target: _PathLike) -> Path:
        self.extract(target)
        target = Path(target) / "libtorch"
        if not target.is_dir():
            raise BuildError("Failed to place RAI dependency: `libtorch`")
        return target


@t.final
class _PTArchiveLinux(_PTArchive):
    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        return ((OperatingSystem.LINUX, Architecture.X64),)

    @property
    def url(self) -> str:
        if self.device == Device.GPU:
            pt_build = "cu117"
        else:
            pt_build = Device.CPU.value
        # pylint: disable-next=line-too-long
        libtorch_archive = (
            f"libtorch-cxx11-abi-shared-without-deps-{self.version}%2B{pt_build}.zip"
        )
        return f"https://download.pytorch.org/libtorch/{pt_build}/{libtorch_archive}"


@t.final
class _PTArchiveMacOSX(_PTArchive):
    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        return (
            (OperatingSystem.DARWIN, Architecture.ARM64),
            (OperatingSystem.DARWIN, Architecture.X64),
        )

    @property
    def url(self) -> str:
        if self.device == Device.GPU:
            raise BuildError("RedisAI does not currently support GPU on Mac OSX")
        if self.architecture == Architecture.X64:
            pt_build = Device.CPU.value
            libtorch_archive = f"libtorch-macos-{self.version}.zip"
            root_url = "https://download.pytorch.org/libtorch"
            return f"{root_url}/{pt_build}/{libtorch_archive}"
        if self.architecture == Architecture.ARM64:
            libtorch_archive = f"libtorch-macos-arm64-{self.version}.zip"
            # pylint: disable-next=line-too-long
            root_url = (
                "https://github.com/CrayLabs/ml_lib_builder/releases/download/v0.1/"
            )
            return f"{root_url}/{libtorch_archive}"

        raise BuildError(f"Unsupported architecture for Pytorch: {self.architecture}")


def _choose_pt_variant(
    os_: OperatingSystem,
) -> t.Union[t.Type[_PTArchiveLinux], t.Type[_PTArchiveMacOSX]]:
    if os_ == OperatingSystem.DARWIN:
        return _PTArchiveMacOSX
    if os_ == OperatingSystem.LINUX:
        return _PTArchiveLinux

    raise BuildError(f"Unsupported OS for PyTorch: {os_}")


@t.final
@dataclass(frozen=True)
class _TFArchive(_WebTGZ, _RAIBuildDependency):
    os_: OperatingSystem
    architecture: Architecture
    device: Device
    version: str

    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        return (
            (OperatingSystem.LINUX, Architecture.X64),
            (OperatingSystem.DARWIN, Architecture.X64),
        )

    @property
    def url(self) -> str:
        if self.architecture == Architecture.X64:
            tf_arch = "x86_64"
        else:
            raise BuildError(
                f"Unexpected Architecture for TF Archive: {self.architecture}"
            )

        if self.os_ == OperatingSystem.LINUX:
            tf_os = "linux"
            tf_device = self.device
        elif self.os_ == OperatingSystem.DARWIN:
            tf_os = "darwin"
            if self.device == Device.GPU:
                raise BuildError("RedisAI does not currently support GPU on Macos")
            tf_device = Device.CPU
        else:
            raise BuildError(f"Unexpected OS for TF Archive: {self.os_}")
        return (
            "https://storage.googleapis.com/tensorflow/libtensorflow/"
            f"libtensorflow-{tf_device.value}-{tf_os}-{tf_arch}-{self.version}.tar.gz"
        )

    @property
    def __rai_dependency_name__(self) -> str:
        return f"libtensorflow@{self.url}"

    def __place_for_rai__(self, target: _PathLike) -> Path:
        target = Path(target) / "libtensorflow"
        target.mkdir()
        self.extract(target)
        return target


@t.final
@dataclass(frozen=True)
class _ORTArchive(_WebTGZ, _RAIBuildDependency):
    os_: OperatingSystem
    device: Device
    version: str

    @staticmethod
    def supported_platforms() -> t.Sequence[t.Tuple[OperatingSystem, Architecture]]:
        return (
            (OperatingSystem.LINUX, Architecture.X64),
            (OperatingSystem.DARWIN, Architecture.X64),
        )

    @property
    def url(self) -> str:
        ort_url_base = (
            "https://github.com/microsoft/onnxruntime/releases/"
            f"download/v{self.version}"
        )
        if self.os_ == OperatingSystem.LINUX:
            ort_os = "linux"
            ort_arch = "x64"
            ort_build = "-gpu" if self.device == Device.GPU else ""
        elif self.os_ == OperatingSystem.DARWIN:
            ort_os = "osx"
            ort_arch = "x86_64"
            ort_build = ""
            if self.device == Device.GPU:
                raise BuildError("RedisAI does not currently support GPU on Macos")
        else:
            raise BuildError(f"Unexpected OS for TF Archive: {self.os_}")
        ort_archive = f"onnxruntime-{ort_os}-{ort_arch}{ort_build}-{self.version}.tgz"
        return f"{ort_url_base}/{ort_archive}"

    @property
    def __rai_dependency_name__(self) -> str:
        return f"onnxruntime@{self.url}"

    def __place_for_rai__(self, target: _PathLike) -> Path:
        target = Path(target).resolve() / "onnxruntime"
        self.extract(target)
        try:
            (extracted_dir,) = target.iterdir()
        except ValueError:
            raise BuildError(
                "Unexpected number of files extracted from ORT archive"
            ) from None
        for file in extracted_dir.iterdir():
            file.rename(target / file.name)
        extracted_dir.rmdir()
        return target


def _git(*args: str) -> None:
    git = Builder.binary_path("git")
    cmd = (git,) + args
    with subprocess.Popen(cmd) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise BuildError(
                f"Command `{' '.join(cmd)}` failed with exit code {proc.returncode}"
            )


def config_git_command(plat: Platform, cmd: t.Sequence[str]) -> t.List[str]:
    """Modify git commands to include autocrlf when on a platform that needs
    autocrlf enabled to behave correctly
    """
    cmd = list(cmd)
    where = next((i for i, tok in enumerate(cmd) if tok.endswith("git")), len(cmd)) + 2
    if where >= len(cmd):
        raise ValueError(f"Failed to locate git command in '{' '.join(cmd)}'")
    if plat == Platform(OperatingSystem.DARWIN, Architecture.ARM64):
        cmd = (
            cmd[:where]
            + ["--config", "core.autocrlf=false", "--config", "core.eol=lf"]
            + cmd[where:]
        )
    return cmd


def _modify_source_files(
    files: t.Union[_PathLike, t.Iterable[_PathLike]], regex: str, replacement: str
) -> None:
    compiled_regex = re.compile(regex)
    with fileinput.input(files=files, inplace=True) as handles:
        for line in handles:
            line = compiled_regex.sub(replacement, line)
            print(line, end="")
