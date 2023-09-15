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

import importlib
import shutil
import subprocess as sp
import sys
import typing as t
from argparse import ArgumentParser, Namespace
from pathlib import Path

from smartsim._core._install.buildenv import SetupError
from smartsim._core._install.builder import BuildError
from smartsim._core.utils import colorize
from smartsim.log import get_logger

SMART_LOGGER_FORMAT = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)


def get_install_path() -> Path:
    module_spec = importlib.util.find_spec("smartsim")

    if module_spec is None:
        raise SetupError("Could not import SmartSim") from None

    # find the path to the setup script
    package_dir = ""
    if module_spec.submodule_search_locations:
        package_dir = module_spec.submodule_search_locations[0]
    package_path = Path(package_dir).resolve()
    if not package_path.is_dir():
        raise SetupError("Could not find SmartSim installation site")

    return package_path


def color_bool(trigger: bool = True) -> str:
    _color = "green" if trigger else "red"
    return colorize(str(trigger), color=_color)


def pip(*args: str) -> None:
    cmd = (sys.executable, "-m", "pip") + args
    with sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE) as proc:
        _, err = proc.communicate()
        if int(proc.returncode) != 0:
            raise BuildError(
                f"`pip` returned with a non-zero exit code:\n{err.decode('utf-8')}"
            )


def clean(core_path: Path, _all: bool = False) -> int:
    """Remove pre existing installations of ML runtimes

    :param _all: Remove all non-python dependencies
    :type _all: bool, optional
    """

    build_temp = core_path / ".third-party"
    if build_temp.is_dir():
        shutil.rmtree(build_temp, ignore_errors=True)

    lib_path = core_path / "lib"
    if lib_path.is_dir():
        # remove RedisAI
        rai_path = lib_path / "redisai.so"
        if rai_path.is_file():
            rai_path.unlink()
            logger.info("Successfully removed existing RedisAI installation")

        backend_path = lib_path / "backends"
        if backend_path.is_dir():
            shutil.rmtree(backend_path, ignore_errors=True)
            logger.info("Successfully removed ML runtimes")

    bin_path = core_path / "bin"
    if bin_path.is_dir() and _all:
        files_to_remove = ["redis-server", "redis-cli", "keydb-server", "keydb-cli"]
        removed = False
        for _file in files_to_remove:
            file_path = bin_path.joinpath(_file)

            if file_path.is_file():
                removed = True
                file_path.unlink()
        if removed:
            logger.info("Successfully removed SmartSim database installation")

    return 0


def get_db_path() -> t.Optional[Path]:
    bin_path = get_install_path() / "_core" / "bin"
    for option in bin_path.iterdir():
        if option.name in ("redis-cli", "keydb-cli"):
            return option
    return None


_CliHandler = t.Callable[[Namespace], int]
_CliParseConfigurator = t.Callable[[ArgumentParser], None]


class MenuItemConfig:
    def __init__(
        self,
        cmd: str,
        description: str,
        handler: _CliHandler,
        configurator: t.Optional[_CliParseConfigurator] = None,
    ):
        self.command = cmd
        self.description = description
        self.handler = handler
        self.configurator = configurator
