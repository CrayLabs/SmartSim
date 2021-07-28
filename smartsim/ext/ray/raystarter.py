import argparse
import os
import shlex
from subprocess import PIPE, STDOUT, Popen

os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser(description="SmartSim Ray head launcher")
parser.add_argument(
    "--port", type=int, help="Port used by Ray to start the Redis server at"
)
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster")
parser.add_argument("--ray-args", type=str, help="Additional arguments to start Ray")
parser.add_argument("--dashboard-port", type=str, help="Ray dashboard port")
args = parser.parse_args()


def current_ip():
    import socket

    hostname = socket.getfqdn(socket.gethostname())
    return socket.gethostbyname(hostname)


print(args)

cliargs = [
    "ray",
    "start",
    "--head",
    "--block",
    f"--port={args.port}",
    f"--redis-password={args.redis_password}",
    f"--node-ip-address={current_ip()}",
    f"--dashboard-port={args.dashboard_port}",
    args.ray_args.strip("\"'"),
]


print(" ".join(cliargs))

p = Popen(cliargs, stdout=PIPE, stderr=STDOUT)

for line in iter(p.stdout.readline, b""):
    print(line.decode("utf-8").rstrip(), flush=True)
