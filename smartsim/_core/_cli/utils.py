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

import subprocess
import typing as t
from pathlib import Path

from smartsim._core._install.buildenv import SetupError
from smartsim._core._install.builder import BuildError
from smartsim._core.utils import colorize
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)


def get_install_path() -> Path:
    try:
        import smartsim as _
    except (ImportError, ModuleNotFoundError):
        raise SetupError("Could not import SmartSim") from None

    # find the path to the setup script
    package_path = Path(_.__path__[0]).resolve()
    if not package_path.is_dir():
        raise SetupError("Could not find SmartSim installation site")

    return package_path


def color_bool(trigger: bool = True) -> str:
    _color = "green" if trigger else "red"
    return colorize(str(trigger), color=_color)


def pip_install(packages: t.List[str], end_point: t.Optional[str] = None, verbose: bool = False) -> None:
    """Install a pip package to be used in the SmartSim build
    Currently only Torch shared libraries are re-used for the build
    """
    # form pip install command
    cmd = ["python", "-m", "pip", "install"]
    cmd.extend(packages)
    if end_point:
        cmd.extend(["-f", end_point])

    cmd_arg = " ".join(cmd)

    if verbose:
        logger.info(f"Installing packages {packages}...")
    proc = subprocess.Popen(
        cmd_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, err = proc.communicate()
    returncode = int(proc.returncode)
    if returncode != 0:
        error = f"{packages} installation failed with exitcode {returncode}\n"
        error += err.decode("utf-8")
        raise BuildError(error)
    if verbose:
        logger.info(f"{packages} installed successfully")
