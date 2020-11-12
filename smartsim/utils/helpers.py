"""
A file of helper functions for SmartSim
"""
import os
from shutil import which
from os import environ

from ..error import SSConfigError


def expand_exe_path(exe):
    """Takes an executable and returns the full path to that executable

    :param exe: exectable or file
    :type exe: str
    """

    # which returns none if not found
    in_path = which(exe)
    if not in_path:
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return os.path.abspath(exe)
        if os.path.isfile(exe) and not os.access(exe, os.X_OK):
            raise SSConfigError(f"File, {exe}, is not an executable")
        else:
            raise SSConfigError(f"Could not locate executable {exe}")
    else:
        return os.path.abspath(in_path)


def get_env(env_var):
    """Retrieve an environment variable through os.environ

    :param str env_var: environment variable to retrieve
    :throws: SSConfigError
    """
    try:
        value = environ[env_var]
        return value
    except KeyError:
        raise SSConfigError("SmartSim environment not set up!")


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
