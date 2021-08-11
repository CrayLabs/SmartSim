import argparse
import os
import socket
from subprocess import PIPE, STDOUT, Popen

import psutil


def get_lb_interface_name():
    """Use psutil to get loopback interface name"""
    net_if_addrs = list(psutil.net_if_addrs())
    for interface in net_if_addrs:
        if interface.startswith("lo"):
            return interface
    raise OSError("Could not find loopback interface name")


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


os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser(
    prefix_chars="+", description="SmartSim Process Launcher"
)
parser.add_argument("+ifname", type=str, help="Network Interface name", default="lo")
parser.add_argument("+command", nargs="+", help="Command to run")
args = parser.parse_args()


def current_ip(interface="lo"):
    if interface == "lo":
        loopback = get_lb_interface_name()
        return get_ip_from_interface(loopback)
    else:
        return get_ip_from_interface(interface)


IP_ADDRESS = current_ip(args.ifname)
COMMAND = args.command + [f"--bind {IP_ADDRESS}"]

print("-" * 10, "  Running  Command  ", "-" * 10, "\n")
print(f"COMMAND: {' '.join(COMMAND)}\n")
print(f"IPADDRESS: {IP_ADDRESS}\n")
print(f"NETWORK: {args.ifname}\n")
print("-" * 30, "\n\n")

print("-" * 10, "  Output  ", "-" * 10, "\n\n")

p = Popen(COMMAND, stdout=PIPE, stderr=STDOUT)

for line in iter(p.stdout.readline, b""):
    print(line.decode("utf-8").rstrip(), flush=True)
