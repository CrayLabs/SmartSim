import argparse
import os
from subprocess import PIPE, STDOUT, Popen

import psutil

from smartsim._core.utils.helpers import get_ip_from_interface
from smartsim.exp.ray import parse_ray_head_node_address


def get_lb_interface_name():
    """Use psutil to get loopback interface name"""
    net_if_addrs = list(psutil.net_if_addrs())
    for interface in net_if_addrs:
        if interface.startswith("lo"):
            return interface
    raise OSError("Could not find loopback interface name")


os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser(
    prefix_chars="+", description="SmartSim Ray head launcher"
)
parser.add_argument(
    "+port", type=int, help="Port used by Ray to start the Redis server at"
)
parser.add_argument("+head", action="store_true")
parser.add_argument("+redis-password", type=str, help="Password of Redis cluster")
parser.add_argument(
    "+ray-args", action="append", help="Additional arguments to start Ray"
)
parser.add_argument("+dashboard-port", type=str, help="Ray dashboard port")
parser.add_argument("+ray-exe", type=str, help="Ray executable", default="ray")
parser.add_argument("+ifname", type=str, help="Interface name", default="lo")
parser.add_argument("+head-log", type=str, help="Head node log")
args = parser.parse_args()

if not args.head and not args.head_log:
    raise argparse.ArgumentError(
        "Ray starter needs +head or +head-log to start head or worker nodes respectively"
    )


def current_ip(interface="lo"):
    if interface == "lo":
        loopback = get_lb_interface_name()
        return get_ip_from_interface(loopback)
    else:
        return get_ip_from_interface(interface)


RAY_IP = current_ip(args.ifname)

cliargs = [
    args.ray_exe,
    "start",
    "--head"
    if args.head
    else f"--address={parse_ray_head_node_address(args.head_log)}:{args.port}",
    "--block",
    f"--node-ip-address={RAY_IP}",
]

if args.ray_args:
    cliargs += args.ray_args
    if args.head and not any(
        [arg.startswith("--dashboard-host") for arg in args.ray_args]
    ):
        cliargs += [f"--dashboard-host={RAY_IP}"]

if args.redis_password:
    cliargs += [f"--redis-password={args.redis_password}"]

if args.head:
    cliargs += [f"--port={args.port}", f"--dashboard-port={args.dashboard_port}"]


cmd = " ".join(cliargs)
print(f"Ray Command: {cmd}")

p = Popen(cliargs, stdout=PIPE, stderr=STDOUT)

for line in iter(p.stdout.readline, b""):
    print(line.decode("utf-8").rstrip(), flush=True)
