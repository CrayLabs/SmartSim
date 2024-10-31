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

import os
import re
import shutil
import stat
import subprocess
import typing as t
from pathlib import Path
from subprocess import SubprocessError

from smartsim._core._install.platform import Architecture, OperatingSystem, Platform
from smartsim._core._install.utils import retrieve
from smartsim._core.utils import expand_exe_path

if t.TYPE_CHECKING:
    from typing_extensions import Never

# TODO: check cmake version and use system if possible to avoid conflicts

_PathLike = t.Union[str, "os.PathLike[str]"]
_T = t.TypeVar("_T")
_U = t.TypeVar("_U")


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
        verbose: bool = False,
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
