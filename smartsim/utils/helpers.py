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
import socket
from os import environ
from shutil import which

import psutil

from ..error import SSConfigError


def get_ip_from_interface(interface):
    """Get IPV4 address of a network interface

    :param interface: interface name
    :type interface: str
    :raises ValueError: if the interface does not exist
    :raises ValueError: if interface does not have an IPV4 address
    :return: ip address of interface
    :rtype: str
    """
    net_if_addrs = psutil.net_if_addrs()
    if interface not in net_if_addrs:

        available = list(net_if_addrs.keys())
        raise ValueError(
            f"{interface} is not a valid network interface. "
            f"Valid network interfaces are: {available}"
        )

    for info in net_if_addrs[interface]:
        if info.family == socket.AF_INET:
            return info.address
    raise ValueError(f"interface {interface} doesn't have an IPv4 address")


def init_default(default, init_value, expected_type=None):
    if init_value is None:
        return default
    if expected_type is not None and not isinstance(init_value, expected_type):
        raise TypeError(f"Argument was of type {type(init_value)}, not {expected_type}")
    return init_value


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
        raise SSConfigError(f"Could not locate executable {exe}")
    return os.path.abspath(in_path)


def get_env(env_var):
    """Retrieve an environment variable through os.environ

    :param str env_var: environment variable to retrieve
    :throws: SSConfigError
    """
    try:
        value = environ[env_var]
        return value
    except KeyError as e:
        raise SSConfigError("SmartSim environment not set up!") from e


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
