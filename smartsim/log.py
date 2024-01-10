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

from __future__ import annotations  # TODO: rm this

import functools
import logging
import os
import pathlib
import sys
import threading
import typing as t

import coloredlogs
from contextvars import ContextVar, copy_context


# constants
DEFAULT_DATE_FORMAT: t.Final[str] = "%H:%M:%S"
DEFAULT_LOG_FORMAT: t.Final[
    str
] = "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
EXPERIMENT_LOG_FORMAT = DEFAULT_LOG_FORMAT.replace("s[%", "s {%(exp_path)s} [%")

# configure colored loggs
coloredlogs.DEFAULT_DATE_FORMAT = DEFAULT_DATE_FORMAT
coloredlogs.DEFAULT_LOG_FORMAT = DEFAULT_LOG_FORMAT


ctx_logger_name = ContextVar("logger_name", default="SmartSim")
ctx_exp_path = ContextVar("exp_path", default="")
ctx_fh_registered = ContextVar("fh_reg", default=False)


def _get_log_level() -> str:
    """Get the logging level based on environment variable
       SMARTSIM_LOG_LEVEL.  If not set, default to info.

       Logging levels
         - quiet: Just shows errors and warnings
         - info: Show basic information and errors (default)
         - debug: Shows info, errors and user debug information
         - developer: Shows everything happening during execution
                      extremely verbose logging.

    :return: Log level for coloredlogs
    :rtype: str
    """
    log_level = os.environ.get("SMARTSIM_LOG_LEVEL", "info").lower()
    if log_level == "quiet":
        return "warning"
    if log_level == "info":
        return "info"
    if log_level == "debug":
        return "debug"
    # extremely verbose logging used internally
    if log_level == "developer":
        return "debug"
    return "info"


class ContextThread(threading.Thread):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self.ctx = copy_context()
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        return self.ctx.run(super().run)


class ContextInjectingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.exp_path = ctx_exp_path.get()
        return True


class ContextAwareLogger(logging.Logger):
    def _log(
        self,
        level: int,
        msg: object,
        args: t.Any,
        exc_info: t.Optional[t.Any] = None,
        extra: t.Optional[t.Any] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """Automatically attach file handlers if contextual information is found"""
        if _exp_path := ctx_exp_path.get():
            file_out = str(pathlib.Path(_exp_path) / "smartsim.out")
            file_err = str(pathlib.Path(_exp_path) / "smartsim.err")

            _lvl = logging.getLevelName(self.level)
            fmt = EXPERIMENT_LOG_FORMAT

            low_pass = LowPassFilter(_lvl)
            h_out = log_to_file(file_out, _lvl, self, fmt, low_pass)
            h_err = log_to_file(file_err, "WARN", self, fmt)

            super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

            for handler in [h_out, h_err]:
                self.removeHandler(handler)

            return

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


def get_logger(
    name: str, log_level: t.Optional[str] = None, fmt: t.Optional[str] = None
) -> logging.Logger:
    """Return a logger instance

    levels:
        - quiet
        - info
        - debug
        - developer

    examples:
        # returns a logger with the name of the module
        logger = get_logger(__name__)

        logger.info("This is a message")
        logger.debug("This is a debug message")
        logger.error("This is an error message")
        logger.warning("This is a warning message")

    :param name: the name of the desired logger
    :type name: str
    :param log_level: what level to set the logger to
    :type log_level: str
    :param fmt: the format of the log messages
    :type fmt: str
    :returns: logger instance
    :rtype: logging.Logger
    """
    # if name is None, then logger is the root logger
    # if not root logger, get the name of file without prefix.
    user_log_level = _get_log_level()
    if user_log_level != "developer":
        name = "SmartSim"

    logging.setLoggerClass(ContextAwareLogger)
    logger = logging.getLogger(name)

    logger.addFilter(ContextInjectingLogFilter())

    if log_level:
        logger.setLevel(log_level)
    else:
        log_level = user_log_level
    coloredlogs.install(level=log_level, logger=logger, fmt=fmt, stream=sys.stdout)

    return logger


class LowPassFilter(logging.Filter):
    """A filter that passes all records below a specified level"""

    def __init__(self, maximum_level: str = "INFO"):
        """Create a low-pass log filter allowing messages below a specific log level

        :param maximum_level: The maximum log level to be passed by the filter
        :type maximum_level: str
        """
        super().__init__()
        self.max = maximum_level

    def filter(self, record: logging.LogRecord) -> bool:
        # If a string representation of the level is passed in,
        # the corresponding numeric value is returned.
        if record.levelno <= logging.getLevelName(self.max):
            return True
        return False


def log_to_file(
    filename: str,
    log_level: str = "warn",
    logger: t.Optional[logging.Logger] = None,
    fmt: t.Optional[str] = None,
    log_filter: t.Optional[logging.Filter] = None,
) -> logging.Handler:
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param filename: the name of the desired log file.
    :type filename: str

    :param log_level: as defined in get_logger.  Can be specified
                      to allow the file to store more or less verbose
                      logging information.
    :type log_level: int | str
    :return: strategy function
    :rtype: callable
    """
    if logger is None:
        logger = logging.getLogger("SmartSim")

    # ensure logs can are written if specified dir doesn't exist
    log_path = pathlib.Path(filename)
    if not log_path.parent.exists():
        filename = log_path.name

    handler = logging.FileHandler(filename, mode="a+", encoding="utf-8")

    if log_filter:
        handler.addFilter(log_filter)

    fmt = fmt or EXPERIMENT_LOG_FORMAT
    formatter = logging.Formatter(fmt=fmt, datefmt=DEFAULT_DATE_FORMAT)

    handler.setFormatter(formatter)
    handler.setLevel(log_level.upper())

    logger.addHandler(handler)
    return handler


#########################################################
# TODO: Move these!!
#########################################################

_T = t.TypeVar("_T")
_RT = t.TypeVar("_RT")
# _ContextT = t.TypeVar("_ContextT")

if t.TYPE_CHECKING:
    from typing_extensions import ParamSpec, Concatenate

    _PR = ParamSpec("_PR")

#########################################################


def contextualize(
    fn: t.Callable[Concatenate[_T, _PR], _RT]
) -> t.Callable[Concatenate[_T, _PR], _RT]:
    def _alter_context(*args: t.Any, **kwargs: t.Any) -> _RT:
        self = args[0]
        exp_path: str = getattr(self, "exp_path", "")
        ctx_exp_path.set(exp_path)
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    def _inner(self: _T, *args: _PR.args, **kwargs: _PR.kwargs) -> _RT:
        ctx = copy_context()
        return ctx.run(_alter_context, self, *args, **kwargs)

    return _inner

# def method_contextualizer(
#     # ctx_var: ContextVar[_ContextT],
#     ctx_map: t.Callable[[_T], _ContextT],
# ) -> t.Callable[
#     [t.Callable[Concatenate[_T, _PR], _RT]],
#     t.Callable[Concatenate[_T, _PR], _RT],
# ]:
#     def _contextualize(
#         fn: t.Callable[Concatenate[_T, _PR], _RT], /
#     ) -> t.Callable[Concatenate[_T, _PR], _RT]:

#         @functools.wraps(fn)
#         def _contextual(
#             self: _T,
#             *args: _PR.args,
#             **kwargs: _PR.kwargs,
#         ) -> _RT:
#             # ctx_val = ctx_map(self)
#             # ctx_val = self.exp_path
#             def _ctx_modifier() -> _RT:
#                 """Thin wrapper to ensure the target method changes
#                 context var after ctx.run"""
#                 # ctx_val = ctx_map(self)
#                 # token = ctx_var.set(ctx_val)
#                 token = ctx_exp_path.set(self.exp_path)
#                 result = fn(self, *args, **kwargs)
#                 # ctx_var.reset(token)
#                 ctx_exp_path.reset(token)
#                 return result

#             ctx = copy_context()
#             return ctx.run(_ctx_modifier)

#         return _contextual

#     return _contextualize
