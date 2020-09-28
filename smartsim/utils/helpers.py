
"""
A file of helper functions for SmartSim
"""
import os
from shutil import which
from os import environ, unsetenv

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
            return SSConfigError(f"File, {exe}, is not an executable")
        else:
            return SSConfigError(f"Could not locate executable {exe}")
    else:
        return in_path

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


def get_config(param, aux, none_ok=False):
    """Search for a configuration parameter in the initialization
        kwargs of an object Also search through an auxiliry dictionary
        in some cases.

        :param str param: parameter to search for
        :param dict aux: auxiliry dictionary to search through (default=None)
        :param bool none_ok: ok to return none if param is not present (default=False)
        :raises KeyError:
        :returns: param if present
    """
    if param in aux.keys():
        return aux[param]
    else:
        if none_ok:
            return None
        else:
            raise KeyError(param)


def remove_env(env_var):
    """Remove a variable from the environment.

    :param env_var: variable to remote
    :type env_var: string
    """

    try:
        unsetenv(env_var)
        del environ[env_var]
        return
    except KeyError:
        return

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
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
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)