# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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
import os
import uuid
from shutil import which
from functools import lru_cache

def create_lockfile_name():
    """Generate a unique lock filename using UUID"""
    name = uuid.uuid4()
    return f"{name}.lock"


@lru_cache(maxsize=20, typed=False)
def check_dev_log_level():
    try:
        lvl = os.environ["SMARTSIM_LOG_LEVEL"]
        if lvl == "developer":
            return True
        return False
    except KeyError:
        return False


def get_base_36_repr(positive_int):
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


def init_default(default, init_value, expected_type=None):
    if init_value is None:
        return default
    if expected_type is not None and not isinstance(init_value, expected_type):
        raise TypeError(f"Argument was of type {type(init_value)}, not {expected_type}")
    return init_value


def expand_exe_path(exe):
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


def is_valid_cmd(command):
    try:
        expand_exe_path(command)
        return True
    except (TypeError, FileNotFoundError):
        return False


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
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
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def delete_elements(dictionary, key_list):
    """Delete elements from a dictionary.
    :param dictionary: the dictionary from which the elements must be deleted.
    :type dictionary: dict
    :param key_list: the list of keys to delete from the dictionary.
    :type key: any
    """
    for key in key_list:
        if key in dictionary:
            del dictionary[key]


def cat_arg_and_value(arg_name, value):
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
        return "=".join((arg_name, str(value)))
    elif arg_name.startswith("-"):
        return " ".join((arg_name, str(value)))
    elif len(arg_name) == 1:
        return " ".join(("-" + arg_name, str(value)))
    else:
        return "=".join(("--" + arg_name, str(value)))
