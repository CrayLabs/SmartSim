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

import functools
import logging
import pathlib
import socket
import sys
import threading
import typing as t
from contextvars import ContextVar, copy_context

import coloredlogs

from smartsim._core.config import CONFIG

# constants
DEFAULT_DATE_FORMAT: t.Final[str] = "%H:%M:%S"
DEFAULT_LOG_FORMAT: t.Final[str] = (
    "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
)
EXPERIMENT_LOG_FORMAT = DEFAULT_LOG_FORMAT.replace("s[%", "s {%(exp_path)s} [%")

# configure colored loggs
coloredlogs.DEFAULT_DATE_FORMAT = DEFAULT_DATE_FORMAT
coloredlogs.DEFAULT_LOG_FORMAT = DEFAULT_LOG_FORMAT

# create context vars used by loggers
ctx_exp_path = ContextVar("exp_path", default="")


# Generic types for method contextualizers
_T = t.TypeVar("_T")
_RT = t.TypeVar("_RT")
_ContextT = t.TypeVar("_ContextT")

if t.TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec

    _PR = ParamSpec("_PR")


def _translate_log_level(user_log_level: str = "info") -> str:
    """Translate value of CONFIG.log_level to one
    accepted as ``level`` option by Python's logging module.

       Logging levels
         - quiet: Just shows errors and warnings
         - info: Show basic information and errors (default)
         - debug: Shows info, errors and user debug information
         - developer: Shows everything happening during execution
                      extremely verbose logging.

    :param user_log_level: log level specified by user, defaults to info
    :returns: Log level for coloredlogs
    """
    user_log_level = user_log_level.lower()
    if user_log_level in ["info", "debug", "warning"]:
        return user_log_level
    if user_log_level == "quiet":
        return "warning"
    # extremely verbose logging used internally
    if user_log_level == "developer":
        return "debug"
    return "info"


def get_exp_log_paths() -> t.Tuple[t.Optional[pathlib.Path], t.Optional[pathlib.Path]]:
    """Returns the output and error file paths to experiment logs.
    Returns None for both paths if experiment context is unavailable.

    :returns: 2-tuple of paths to experiment logs in form (output_path, error_path)
    """
    default_paths = None, None

    if _path := ctx_exp_path.get():
        file_out = pathlib.Path(_path) / CONFIG.telemetry_subdir / "logs/smartsim.out"
        file_err = pathlib.Path(_path) / CONFIG.telemetry_subdir / "logs/smartsim.err"
        return file_out, file_err

    return default_paths


class ContextThread(threading.Thread):
    """Thread that ensures the context vars of the caller are available"""

    def run(self) -> None:
        """Execute a thread on a copy of the current thread context"""
        ctx = copy_context()
        return ctx.run(super().run)


class ContextInjectingLogFilter(logging.Filter):
    """Filter that performs enrichment of a log record by adding context
    information about the experiment being executed"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich log records with active experiment context

        :param record: the record to evaluate for filtering
        :returns: always True
        """
        record.exp_path = ctx_exp_path.get()
        return True


class HostnameFilter(logging.Filter):
    """Filter that performs enrichment of a log record by adding
    the hostname of the machine executing the code"""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._hostname = ""

    @property
    @functools.lru_cache
    def hostname(self) -> str:
        """Returns the hostname of the machine executing the code"""
        self._hostname = socket.gethostname()
        return self._hostname

    def filter(self, record: logging.LogRecord) -> bool:
        # the hostname may already added if using the `ColoredLogs` plugin
        if not hasattr(record, "hostname"):
            record.hostname = self.hostname
        return True


class ContextAwareLogger(logging.Logger):
    """A logger customized to automatically write experiment logs to a
    dynamic target directory by inspecting the value of a context var"""

    def __init__(self, name: str, level: t.Union[int, str] = 0) -> None:
        super().__init__(name, level)
        self.addFilter(ContextInjectingLogFilter(name="exp-ctx-log-filter"))

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
        file_out, file_err = get_exp_log_paths()

        if not all([file_out, file_err]):
            super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
            return

        _lvl = logging.getLevelName(self.level)
        fmt = EXPERIMENT_LOG_FORMAT

        low_pass = LowPassFilter(_lvl)
        h_out = log_to_exp_file(str(file_out), self, _lvl, fmt, low_pass)
        h_err = log_to_exp_file(str(file_err), self, "WARN", fmt)

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

        for handler in [h_out, h_err]:
            self.removeHandler(handler)


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
    :param log_level: what level to set the logger to
    :param fmt: the format of the log messages
    :returns: logger instance
    """
    # if name is None, then logger is the root logger
    # if not root logger, get the name of file without prefix.
    user_log_level = CONFIG.log_level
    if user_log_level != "developer":
        name = "SmartSim"

    logging.setLoggerClass(ContextAwareLogger)
    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    else:
        log_level = _translate_log_level(user_log_level)
    coloredlogs.install(level=log_level, logger=logger, fmt=fmt, stream=sys.stdout)
    return logger


class LowPassFilter(logging.Filter):
    """A filter that passes all records below a specified level"""

    def __init__(self, maximum_level: str = "INFO"):
        """Create a low-pass log filter allowing messages below a specific log level

        :param maximum_level: The maximum log level to be passed by the filter
        """
        super().__init__()
        self.max = maximum_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records; pass those less than or equal to the maximum level

        :param record: the record to evaluate for filtering
        :returns: True if record level passes filter, False otherwise
        """
        # If a string representation of the level is passed in,
        # the corresponding numeric value is returned.
        level_no: int = logging.getLevelName(self.max)
        return record.levelno <= level_no


def log_to_file(filename: str, log_level: str = "debug") -> None:
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param filename: the name of the desired log file.
    :param log_level: as defined in get_logger.  Can be specified
                      to allow the file to store more or less verbose
                      logging information.
    """
    logger = logging.getLogger("SmartSim")
    stream = open(  # pylint: disable=consider-using-with
        filename, "w+", encoding="utf-8"
    )
    coloredlogs.install(stream=stream, logger=logger, level=log_level)


def log_to_exp_file(
    filename: str,
    logger: logging.Logger,
    log_level: str = "warn",
    fmt: t.Optional[str] = EXPERIMENT_LOG_FORMAT,
    log_filter: t.Optional[logging.Filter] = None,
) -> logging.Handler:
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param filename: the name of the desired log file.
    :param log_level: as defined in get_logger.  Can be specified
                      to allow the file to store more or less verbose
                      logging information.
    :param logger: an existing logger to add the handler to
    :param fmt: a log format for the handler (otherwise, EXPERIMENT_LOG_FORMAT)
    :param log_filter: log filter to attach to handler
    :return: logging.Handler
    """
    # ensure logs are written even if specified dir doesn't exist
    log_path = pathlib.Path(filename)
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(filename, mode="a+", encoding="utf-8")

    if log_filter:
        handler.addFilter(log_filter)

    formatter = logging.Formatter(fmt=fmt, datefmt=DEFAULT_DATE_FORMAT)

    handler.setFormatter(formatter)
    handler.setLevel(log_level.upper())

    logger.addHandler(handler)
    return handler


def method_contextualizer(
    ctx_var: ContextVar[_ContextT],
    ctx_map: t.Callable[[_T], _ContextT],
) -> """t.Callable[
    [t.Callable[Concatenate[_T, _PR], _RT]],
    t.Callable[Concatenate[_T, _PR], _RT],
]""":
    """Parameterized-decorator factory that enables a target value
    to be placed into global context prior to execution of the
    decorated method.
    Usage Note: the use of `self` below requires that the decorated function is passed
    the object containing a value that will be modified in the context. `ctx_map`
    must accept an instance of matching type.

    :param ctx_var: The ContextVar that will be modified
    :param ctx_map: A function that returns the value to be set to ctx_var
    """

    def _contextualize(
        fn: "t.Callable[Concatenate[_T, _PR], _RT]", /
    ) -> "t.Callable[Concatenate[_T, _PR], _RT]":
        """Executes the decorated method in a cloned context and ensures
        `ctx_var` is updated to the value returned by `ctx_map` prior to
        calling the decorated method"""

        @functools.wraps(fn)
        def _contextual(
            self: _T,
            *args: "_PR.args",
            **kwargs: "_PR.kwargs",
        ) -> _RT:
            """A decorator operator that runs the decorated method in a new
            context with the desired contextual information modified."""

            def _ctx_modifier() -> _RT:
                """Helper to simplify calling the target method with the
                modified value set in `ctx_var`"""
                ctx_val = ctx_map(self)
                token = ctx_var.set(ctx_val)
                result = fn(self, *args, **kwargs)
                ctx_var.reset(token)
                return result

            ctx = copy_context()
            return ctx.run(_ctx_modifier)

        return _contextual

    return _contextualize
