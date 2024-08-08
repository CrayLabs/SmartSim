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
import fileinput
import itertools
import os
import pathlib
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

import git

if t.TYPE_CHECKING:
    from typing_extensions import Never

from smartsim._core.config import CONFIG
from smartsim._core._install.platform import OperatingSystem, Device, Architecture, Platform
from smartsim._core._install.mlpackages import MLPackage
from smartsim._core._install.utils import PackageRetriever



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
        pt_build = self.device.torch_suffix()
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
        if self.device.is_gpu():
            raise BuildError("RedisAI does not currently support GPU on Mac OSX")
        if self.architecture == Architecture.X64:
            pt_build = "cpu"
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
            tf_device = "gpu" if self.device.is_gpu() else "cpu"
        elif self.os_ == OperatingSystem.DARWIN:
            tf_os = "darwin"
            if self.device.is_gpu():
                raise BuildError("RedisAI does not currently support GPU on Macos")
            tf_device = "cpu"
        else:  # pragma: no cover
            _assert_never(self.os_, message=f"Unexpected OS for TF Archive: {self.os_}")
        return (
            "https://storage.googleapis.com/tensorflow/libtensorflow/"
            f"libtensorflow-{tf_device}-{tf_os}-{tf_arch}-{self.version}.tar.gz"
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
            ort_build = "-gpu" if self.device.is_gpu() else ""
        elif self.os_ == OperatingSystem.DARWIN:
            ort_os = "osx"
            ort_arch = "x86_64"
            ort_build = ""
            if self.device.is_gpu():
                raise BuildError("RedisAI does not currently support GPU on Macos")
        else:  # pragma: no cover
            msg = f"Unexpected OS for ONNX Runtime Archive: {self.os_}"
            _assert_never(self.os_, message=msg)
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


def _assert_never(
    obj: "Never", *, message: t.Optional[str] = None
) -> t.NoReturn:  # pragma: no cover
    raise BuildError(
        f"Unexpected value `{repr(obj)}` encountered during build process"
        if message is None
        else message
    )




