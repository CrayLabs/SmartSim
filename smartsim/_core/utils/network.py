# BSD 2-Clause License
#
# Copyright (c) 2021-2023 Hewlett Packard Enterprise
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

import socket

import psutil

"""
A handful of useful functions for dealing with networks
"""


def get_ip_from_host(host: str) -> str:
    """Return the IP address for the interconnect.

    :param host: hostname of the compute node e.g. nid00004
    :type host: str
    :returns: ip of host
    :rtype: str
    """
    ip_address = socket.gethostbyname(host)
    return ip_address


# impossible to cover as it's only used in entrypoints
def get_ip_from_interface(interface: str) -> str:  # pragma: no cover
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


# impossible to cover as it's only used in entrypoints
def get_lb_interface_name() -> str:  # pragma: no cover
    """Use psutil to get loopback interface name"""
    net_if_addrs = list(psutil.net_if_addrs())
    for interface in net_if_addrs:
        if interface.startswith("lo"):
            return interface
    raise OSError("Could not find loopback interface name")


def current_ip(interface: str = "lo") -> str:  # pragma: no cover
    if interface == "lo":
        loopback = get_lb_interface_name()
        return get_ip_from_interface(loopback)

    return get_ip_from_interface(interface)
