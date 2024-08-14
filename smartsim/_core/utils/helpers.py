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

"""
A file of helper functions for SmartSim
"""
from __future__ import annotations

import base64
import collections.abc
import os
import signal
import subprocess
import typing as t
import uuid
from datetime import datetime
from functools import lru_cache
from shutil import which

if t.TYPE_CHECKING:
    from types import FrameType


_T = t.TypeVar("_T")
_HashableT = t.TypeVar("_HashableT", bound=t.Hashable)
_TSignalHandlerFn = t.Callable[[int, t.Optional["FrameType"]], object]


def check_name(name: str) -> None:
    """
    Checks if the input name is valid.

    :param name: The name to be checked.

    :raises ValueError: If the name contains the path separator (os.path.sep).
    """
    if os.path.sep in name:
        raise ValueError("Invalid input: String contains the path separator.")


def unpack_fs_identifier(fs_id: str, token: str) -> t.Tuple[str, str]:
    """Unpack the unformatted feature store identifier
    and format for env variable suffix using the token
    :param fs_id: the unformatted feature store identifier eg. identifier_1
    :param token: character to use to construct the fs suffix
    :return: fs id suffix and formatted fs_id e.g. ("_identifier_1", "identifier_1")
    """

    if fs_id == "featurestore":
        return "", ""
    fs_name_suffix = token + fs_id
    return fs_name_suffix, fs_id


def unpack_colo_fs_identifier(fs_id: str) -> str:
    """Create feature store identifier suffix for colocated feature store

    :param fs_id: the unformatted feature store identifier
    :return: fs suffix
    """
    return "_" + fs_id if fs_id else ""


def create_short_id_str() -> str:
    return str(uuid.uuid4())[:7]


def create_lockfile_name() -> str:
    """Generate a unique lock filename using UUID"""
    lock_suffix = create_short_id_str()
    return f"smartsim-{lock_suffix}.lock"


@lru_cache(maxsize=20, typed=False)
def check_dev_log_level() -> bool:
    lvl = os.environ.get("SMARTSIM_LOG_LEVEL", "")
    return lvl == "developer"


def fmt_dict(value: t.Mapping[str, t.Any]) -> str:
    fmt_str = ""
    for k, v in value.items():
        fmt_str += "\t" + str(k) + " = " + str(v)
        fmt_str += "\n" if k != list(value.keys())[-1] else ""
    return fmt_str


def get_base_36_repr(positive_int: int) -> str:
    """Converts a positive integer to its base 36 representation

    :param positive_int: the positive integer to convert
    :return: base 36 representation of the given positive int
    """
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []

    while positive_int:
        next_digit = digits[positive_int % 36]
        result.append(next_digit)
        positive_int //= 36

    return "".join(reversed(result))


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


def is_valid_cmd(command: t.Union[str, None]) -> bool:
    try:
        if command:
            expand_exe_path(command)
            return True
    except (TypeError, FileNotFoundError):
        return False

    return False


color2num = {
    "gray": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "crimson": 38,
}


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """
    Colorize a string.
    This function was originally written by John Schulman.
    And then borrowed from spinningup
    https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return f"\x1b[{';'.join(attr)}m{string}\x1b[0m"


def delete_elements(dictionary: t.Dict[str, t.Any], key_list: t.List[str]) -> None:
    """Delete elements from a dictionary.
    :param dictionary: the dictionary from which the elements must be deleted.
    :param key_list: the list of keys to delete from the dictionary.
    """
    for key in key_list:
        if key in dictionary:
            del dictionary[key]


def cat_arg_and_value(arg_name: str, value: str) -> str:
    """Concatenate a command line argument and its value

    This function returns ``arg_name`` and ``value
    concatenated in the best possible way for a command
    line execution, namely:
    - if arg_name starts with `--` (e.g. `--arg`):
      `arg_name=value` is returned (i.e. `--arg=val`)
    - if arg_name starts with `-` (e.g. `-a`):
      `arg_name value` is returned (i.e. `-a val`)
    - if arg_name does not start with `-` and it is a
      long option (e.g. `arg`):
      `--arg_name=value` (i.e., `--arg=val`)
    - if arg_name does not start with `-` and it is a
      short option (e.g. `a`):
      `-arg_name=value` (i.e., `-a val`)

    :param arg_name: the command line argument name
    :param value: the command line argument value
    """

    if arg_name.startswith("--"):
        return f"{arg_name}={value}"
    if arg_name.startswith("-"):
        return f"{arg_name} {value}"
    if len(arg_name) == 1:
        return f"-{arg_name} {value}"

    return f"--{arg_name}={value}"


def get_ts_ms() -> int:
    """Return the current timestamp (accurate to milliseconds) cast to an integer"""
    return int(datetime.now().timestamp() * 1000)


def encode_cmd(cmd: t.Sequence[str]) -> str:
    """Transform a standard command list into an encoded string safe for providing as an
    argument to a proxy entrypoint
    """
    if not cmd:
        raise ValueError("Invalid cmd supplied")

    ascii_cmd = "|".join(cmd).encode("ascii")
    encoded_cmd = base64.b64encode(ascii_cmd).decode("ascii")
    return encoded_cmd


def decode_cmd(encoded_cmd: str) -> t.List[str]:
    """Decode an encoded command string to the original command list format"""
    if not encoded_cmd.strip():
        raise ValueError("Invalid cmd supplied")

    decoded_cmd = base64.b64decode(encoded_cmd.encode("ascii"))
    cleaned_cmd = decoded_cmd.decode("ascii").split("|")

    return cleaned_cmd


def check_for_utility(util_name: str) -> str:
    """Check for existence of the provided CLI utility.

    :param util_name: CLI utility to locate
    :returns: Full path to executable if found. Otherwise, empty string"""
    utility = ""

    try:
        utility = expand_exe_path(util_name)
    except FileNotFoundError:
        ...

    return utility


def execute_platform_cmd(cmd: str) -> t.Tuple[str, int]:
    """Execute the platform check command as a subprocess

    :param cmd: the command to execute
    :returns: True if platform is cray ex, False otherwise"""
    process = subprocess.run(
        cmd.split(),
        capture_output=True,
        check=False,
    )
    return process.stdout.decode("utf-8"), process.returncode


class CrayExPlatformResult:
    locate_msg = "Unable to locate `{0}`."

    def __init__(self, ldconfig: t.Optional[str], fi_info: t.Optional[str]) -> None:
        self.ldconfig: t.Optional[str] = ldconfig
        self.fi_info: t.Optional[str] = fi_info
        self.has_pmi: bool = False
        self.has_pmi2: bool = False
        self.has_cxi: bool = False

    @property
    def has_ldconfig(self) -> bool:
        return bool(self.ldconfig)

    @property
    def has_fi_info(self) -> bool:
        return bool(self.fi_info)

    @property
    def is_cray(self) -> bool:
        return all(
            (
                self.has_ldconfig,
                self.has_fi_info,
                self.has_pmi,
                self.has_pmi2,
                self.has_cxi,
            )
        )

    @property
    def failures(self) -> t.List[str]:
        """Return a list of messages describing all failed validations"""
        failure_messages = []

        if not self.has_ldconfig:
            failure_messages.append(self.locate_msg.format("ldconfig"))

        if not self.has_fi_info:
            failure_messages.append(self.locate_msg.format("fi_info"))

        if self.has_ldconfig and self.has_fi_info:
            if not self.has_pmi:
                failure_messages.append(self.locate_msg.format("pmi.so"))
            if not self.has_pmi2:
                failure_messages.append(self.locate_msg.format("pmi2.so"))
            if not self.has_cxi:
                failure_messages.append(self.locate_msg.format("cxi.so"))

        return failure_messages


def check_platform() -> CrayExPlatformResult:
    """Returns True if the current platform is identified as Cray EX and
    HSTA-aware dragon package can be installed, False otherwise.

    :returns: True if current platform is Cray EX, False otherwise"""

    # ldconfig -p | grep cray | grep pmi.so &&
    # ldconfig -p | grep cray | grep pmi2.so &&
    # fi_info | grep cxi

    ldconfig = check_for_utility("ldconfig")
    fi_info = check_for_utility("fi_info")

    result = CrayExPlatformResult(ldconfig, fi_info)
    if not all((result.has_ldconfig, result.has_fi_info)):
        return result

    ldconfig1 = f"{ldconfig} -p"
    ldc_out1, _ = execute_platform_cmd(ldconfig1)
    candidates = [x for x in ldc_out1.split("\n") if "cray" in x]
    result.has_pmi = any(x for x in candidates if "pmi.so" in x)

    ldconfig2 = f"{ldconfig} -p"
    ldc_out2, _ = execute_platform_cmd(ldconfig2)
    candidates = [x for x in ldc_out2.split("\n") if "cray" in x]
    result.has_pmi2 = any(x for x in candidates if "pmi2.so" in x)

    fi_info_out, _ = execute_platform_cmd(fi_info)
    result.has_cxi = any(x for x in fi_info_out.split("\n") if "cxi" in x)

    return result


def is_crayex_platform() -> bool:
    """Returns True if the current platform is identified as Cray EX and
    HSTA-aware dragon package can be installed, False otherwise.

    :returns: True if current platform is Cray EX, False otherwise"""
    result = check_platform()
    return result.is_cray


def first(predicate: t.Callable[[_T], bool], iterable: t.Iterable[_T]) -> _T | None:
    """Return the first instance of an iterable that meets some precondition.
    Any elements of the iterable that do not meet the precondition will be
    forgotten. If no item in the iterable is found that meets the predicate,
    `None` is returned. This is roughly equivalent to

    .. highlight:: python
    .. code-block:: python

        next(filter(predicate, iterable), None)

    but does not require the predicate to be a type guard to type check.

    :param predicate: A function that returns `True` or `False` given a element
                      of the iterable
    :param iterable: An iterable that yields elements to evealuate
    :returns: The first element of the iterable to make the the `predicate`
              return `True`
    """
    return next((item for item in iterable if predicate(item)), None)


def unique(iterable: t.Iterable[_HashableT]) -> t.Iterable[_HashableT]:
    """Iterate over an iterable, yielding only unique values.

    This helper function will maintain a set of seen values in memory and yield
    any values not previously seen during iteration. This is nice if you know
    you will be iterating over the iterable exactly once, but if you need to
    iterate over the iterable multiple times, it would likely use less memory
    to cast the iterable to a set first.

    :param iterable: An iterable of possibly not unique values.
    :returns: An iterable of unique values with order unchanged from the
        original iterable.
    """
    seen = set()
    for item in filter(lambda x: x not in seen, iterable):
        seen.add(item)
        yield item


def group_by(
    fn: t.Callable[[_T], _HashableT], items: t.Iterable[_T]
) -> t.Mapping[_HashableT, t.Collection[_T]]:
    """Iterate over an iterable and group the items based on the return of some
    mapping function. Works similar to SQL's "GROUP BY" statement, but works
    over an arbitrary mapping function.

    :param fn: A function mapping the iterable values to some hashable values
    :items: An iterable yielding items to group by mapping function return.
    :returns: A mapping of mapping function return values to collection of
        items that returned that value when fed to the mapping function.
    """
    groups = collections.defaultdict[_HashableT, list[_T]](list)
    for item in items:
        groups[fn(item)].append(item)
    return dict(groups)


@t.final
class SignalInterceptionStack(collections.abc.Collection[_TSignalHandlerFn]):
    """Registers a stack of callables to be called when a signal is
    received before calling the original signal handler.
    """

    def __init__(
        self,
        signalnum: int,
        callbacks: t.Optional[t.Iterable[_TSignalHandlerFn]] = None,
    ) -> None:
        """Set up a ``SignalInterceptionStack`` for particular signal number.

        .. note::
            This class typically should not be instanced directly as it will
            change the registered signal handler regardless of if a signal
            interception stack is already present. Instead, it is generally
            best to create or get a signal interception stack for a particular
            signal number via the `get` factory method.

        :param signalnum: The signal number to intercept
        :param callbacks: A iterable of functions to call upon receiving the signal
        """
        self._callbacks = list(callbacks) if callbacks else []
        self._original = signal.signal(signalnum, self)

    def __call__(self, signalnum: int, frame: t.Optional["FrameType"]) -> None:
        """Handle the signal on which the interception stack was registered.
        End by calling the originally registered signal hander (if present).

        :param frame: The current stack frame
        """
        for fn in self:
            fn(signalnum, frame)
        if callable(self._original):
            self._original(signalnum, frame)

    def __contains__(self, obj: object) -> bool:
        return obj in self._callbacks

    def __iter__(self) -> t.Iterator[_TSignalHandlerFn]:
        return reversed(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    @classmethod
    def get(cls, signalnum: int) -> "SignalInterceptionStack":
        """Fetch an existing ``SignalInterceptionStack`` or create a new one
        for a particular signal number.

        :param signalnum: The singal number of the signal interception stack
                          should be registered
        :returns: The existing or created signal interception stack
        """
        handler = signal.getsignal(signalnum)
        if isinstance(handler, cls):
            return handler
        return cls(signalnum, [])

    def push(self, fn: _TSignalHandlerFn) -> None:
        """Add a callback to the signal interception stack.

        :param fn: A callable to add to the unique signal stack
        """
        self._callbacks.append(fn)

    def push_unique(self, fn: _TSignalHandlerFn) -> bool:
        """Add a callback to the signal interception stack if and only if the
        callback is not already present.

        :param fn: A callable to add to the unique signal stack
        :returns: True if the callback was added, False if the callback was
                  already present
        """
        if did_push := fn not in self:
            self.push(fn)
        return did_push
