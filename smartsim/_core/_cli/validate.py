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

import argparse
import contextlib
import io
import os
import os.path
import tempfile
import typing as t
from types import TracebackType

from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core._install.platform import Device
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# Many of the functions in this module will import optional
# ML python packages only if they are needed to test the build is working
#
# pylint: disable=import-error,import-outside-toplevel
# mypy: disable-error-code="import"


if t.TYPE_CHECKING:
    # pylint: disable-next=unsubscriptable-object
    _TemporaryDirectory = tempfile.TemporaryDirectory[str]
else:
    _TemporaryDirectory = tempfile.TemporaryDirectory


class _VerificationTempDir(_TemporaryDirectory):
    """A Temporary directory to be used as a context manager that will only
    clean itself up if no error is raised within its context
    """

    def __exit__(
        self,
        exc: t.Optional[t.Type[BaseException]],
        value: t.Optional[BaseException],
        tb: t.Optional[TracebackType],
    ) -> None:
        if not value:  # Yay, no error! Clean up as normal
            super().__exit__(exc, value, tb)
        else:  # Uh-oh! Better make sure this is not implicitly cleaned up
            self._finalizer.detach()  # type: ignore[attr-defined]


def execute(args: argparse.Namespace, _unparsed_args: argparse.Namespace) -> int:
    """Validate the SmartSim installation works as expected given a
    simple experiment
    """
    temp_dir = ""
    device = Device(args.device)
    try:
        with contextlib.ExitStack() as ctx:
            temp_dir = ctx.enter_context(_VerificationTempDir(dir=os.getcwd()))
            validate_env = {
                "SR_LOG_LEVEL": os.environ.get("SR_LOG_LEVEL", "INFO"),
            }
            if device == Device.GPU:
                validate_env["CUDA_VISIBLE_DEVICES"] = "0"
            ctx.enter_context(_env_vars_set_to(validate_env))
    except Exception as e:
        logger.error(
            "SmartSim failed to run a simple experiment!\n"
            f"Experiment failed due to the following exception:\n{e}",
            exc_info=True,
        )
        if temp_dir:
            logger.info(f"Output files are available at `{temp_dir}`")
        return os.EX_SOFTWARE
    return os.EX_OK


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Build the parser for the command"""
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=None,
        help=(
            "The port on which to run the feature store for the mini experiment. "
            "If not provided, `smart` will attempt to automatically select an "
            "open port"
        ),
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        default=Device.CPU.value,
        choices=[device.value for device in Device],
        help="Device to test the ML backends against",
    )


@contextlib.contextmanager
def _env_vars_set_to(
    evars: t.Mapping[str, t.Optional[str]]
) -> t.Generator[None, None, None]:
    envvars = tuple((var, os.environ.pop(var, None), val) for var, val in evars.items())
    for var, _, tmpval in envvars:
        _set_or_del_env_var(var, tmpval)
    try:
        yield
    finally:
        for var, origval, _ in reversed(envvars):
            _set_or_del_env_var(var, origval)


def _set_or_del_env_var(var: str, val: t.Optional[str]) -> None:
    if val is not None:
        os.environ[var] = val
    else:
        os.environ.pop(var, None)
