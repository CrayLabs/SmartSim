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

"""
A file of helper functions for SmartSim
"""
import base64
import os
import typing as t
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from shutil import which

from smartsim._core._install.builder import TRedisAIBackendStr as _TRedisAIBackendStr


def unpack_db_identifier(db_id: str, token: str) -> t.Tuple[str, str]:
    """Unpack the unformatted database identifier
    and format for env variable suffix using the token
    :param db_id: the unformatted database identifier eg. identifier_1
    :type db_id: str
    :param token: character to use to construct the db suffix
    :type token: str
    :return: db id suffix and formatted db_id e.g. ("_identifier_1", "identifier_1")
    :rtype: (str, str)
    """

    if db_id == "orchestrator":
        return "", ""
    db_name_suffix = token + db_id
    return db_name_suffix, db_id


def unpack_colo_db_identifier(db_id: str) -> str:
    """Create database identifier suffix for colocated database
    :param db_id: the unformatted database identifier
    :type db_id: str
    :return: db suffix
    :rtype: str
    """
    return "_" + db_id if db_id else ""


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


def fmt_dict(value: t.Dict[str, t.Any]) -> str:
    fmt_str = ""
    for k, v in value.items():
        fmt_str += "\t" + str(k) + " = " + str(v)
        fmt_str += "\n" if k != list(value.keys())[-1] else ""
    return fmt_str


def get_base_36_repr(positive_int: int) -> str:
    """Converts a positive integer to its base 36 representation
    :param positive_int: the positive integer to convert
    :type positive_int: int
    :return: base 36 representation of the given positive int
    :rtype: str
    """
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []

    while positive_int:
        next_digit = digits[positive_int % 36]
        result.append(next_digit)
        positive_int //= 36

    return "".join(reversed(result))


def init_default(
    default: t.Any,
    init_value: t.Any,
    expected_type: t.Union[t.Type[t.Any], t.Tuple[t.Type[t.Any], ...], None] = None,
) -> t.Any:
    if init_value is None:
        return default
    if expected_type is not None and not isinstance(init_value, expected_type):
        raise TypeError(f"Argument was of type {type(init_value)}, not {expected_type}")
    return init_value


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
    :type dictionary: dict
    :param key_list: the list of keys to delete from the dictionary.
    :type key: any
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
    :type arg_name: str
    :param value: the command line argument value
    :type value: str
    """

    if arg_name.startswith("--"):
        return f"{arg_name}={value}"
    if arg_name.startswith("-"):
        return f"{arg_name} {value}"
    if len(arg_name) == 1:
        return f"-{arg_name} {value}"

    return f"--{arg_name}={value}"


def _installed(base_path: Path, backend: str) -> bool:
    """
    Check if a backend is available for the RedisAI module.
    """
    backend_key = f"redisai_{backend}"
    backend_path = base_path / backend_key / f"{backend_key}.so"
    backend_so = Path(os.environ.get("RAI_PATH", backend_path)).resolve()

    return backend_so.is_file()


def redis_install_base(backends_path: t.Optional[str] = None) -> Path:
    # pylint: disable-next=import-outside-toplevel
    from ..._core.config import CONFIG

    base_path = Path(backends_path) if backends_path else CONFIG.lib_path / "backends"
    return base_path


def installed_redisai_backends(
    backends_path: t.Optional[str] = None,
) -> t.Set[_TRedisAIBackendStr]:
    """Check which ML backends are available for the RedisAI module.

    The optional argument ``backends_path`` is needed if the backends
    have not been built as part of the SmartSim building process (i.e.
    they have not been built by invoking `smart build`). In that case
    ``backends_path`` should point to the directory containing e.g.
    the backend directories (`redisai_tensorflow`, `redisai_torch`,
    `redisai_onnxruntime`, or `redisai_tflite`).

    :param backends_path: path containing backends, defaults to None
    :type backends_path: str, optional
    :return: list of installed RedisAI backends
    :rtype: set[str]
    """
    # import here to avoid circular import
    base_path = redis_install_base(backends_path)
    backends: t.Set[_TRedisAIBackendStr] = {
        "tensorflow",
        "torch",
        "onnxruntime",
        "tflite",
    }

    return {backend for backend in backends if _installed(base_path, backend)}


def get_ts() -> int:
    """Return the current timestamp (accurate to seconds) cast to an integer"""
    return int(datetime.timestamp(datetime.now()))


def encode_cmd(cmd: t.List[str]) -> str:
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
