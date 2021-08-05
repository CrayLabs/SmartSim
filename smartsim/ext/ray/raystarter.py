import argparse
import os
from subprocess import PIPE, STDOUT, Popen

from smartsim.utils.helpers import get_ip_from_interface, get_lb_interface_name

os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser(description="SmartSim Ray head launcher")
parser.add_argument(
    "--port", type=int, help="Port used by Ray to start the Redis server at"
)
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster")
parser.add_argument("--ray-args", type=str, help="Additional arguments to start Ray")
parser.add_argument("--dashboard-port", type=str, help="Ray dashboard port")
parser.add_argument("--ray-exe", type=str, help="Ray executable", default="ray")
parser.add_argument("--ifname", type=str, help="Interface name", default="lo")
args = parser.parse_args()

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
    "--head",
    "--block",
    f"--port={args.port}",
    f"--redis-password={args.redis_password}",
    f"--node-ip-address=0.0.0.0",
    f"--dashboard-port={args.dashboard_port}",
    args.ray_args.strip("\"'")
]

cmd = " ".join(cliargs)
print(f"Ray Command: {cmd}")

p = Popen(cliargs, stdout=PIPE, stderr=STDOUT)

for line in iter(p.stdout.readline, b""):
    print(line.decode("utf-8").rstrip(), flush=True)
