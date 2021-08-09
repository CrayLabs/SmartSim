import argparse
import os
import re
import time
from subprocess import PIPE, STDOUT, Popen

from smartsim.utils.helpers import get_ip_from_interface, get_lb_interface_name


def parse_ray_head_node_address(head_log):
    """Get the ray head node host address from the log file produced
    by the head process.

    :return: address of the head host
    :rtype: str
    """

    max_attempts = 60
    attempts = 0
    while not os.path.isfile(head_log):
        time.sleep(1)
        attempts += 1
        if attempts == max_attempts:
            raise RuntimeError("Could not find Ray cluster head address.")

    attempts = 0
    head_ip = None
    while head_ip is None:
        time.sleep(1)
        with open(head_log) as fp:
            line = fp.readline()
            while line:
                plain_line = re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", line)
                if "Local node IP:" in plain_line:
                    matches = re.search(r"(?<=Local node IP: ).*", plain_line)
                    head_ip = matches.group()
                    break
                line = fp.readline()
        attempts += 1
        if attempts == max_attempts:
            raise RuntimeError("Could not find Ray cluster head address.")

    return head_ip


os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser(description="SmartSim Ray head launcher")
parser.add_argument(
    "--port", type=int, help="Port used by Ray to start the Redis server at"
)
parser.add_argument("--head", action="store_true")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster")
parser.add_argument("--ray-args", type=str, help="Additional arguments to start Ray")
parser.add_argument("--dashboard-port", type=str, help="Ray dashboard port")
parser.add_argument("--ray-exe", type=str, help="Ray executable", default="ray")
parser.add_argument("--ifname", type=str, help="Interface name", default="lo")
parser.add_argument("--head-log", type=str, help="Head node log")
args = parser.parse_args()

if not args.head and not args.head_log:
    raise argparse.ArgumentError(
        "Ray starter needs --head or --head-log to start head or worker nodes respectively"
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
    args.ray_args.strip("\"'"),
    f"--node-ip-address={RAY_IP}",
]

if args.redis_password:
    cliargs += [f"--redis-password={args.redis_password}"]

# On some systems, ssh to compute nodes (and port forwarding) is not allowed.
# If that's the case, the user should bind the dashboard to 0.0.0.0,
# which means it is available from all interfaces.
if args.head:
    cliargs += [f"--port={args.port}", f"--dashboard-port={args.dashboard_port}"]


cmd = " ".join(cliargs)
print(f"Ray Command: {cmd}")

p = Popen(cliargs, stdout=PIPE, stderr=STDOUT)

for line in iter(p.stdout.readline, b""):
    print(line.decode("utf-8").rstrip(), flush=True)
