import argparse
import os
from subprocess import Popen, PIPE, STDOUT

os.environ["PYTHONUNBUFFERED"]="1"

parser = argparse.ArgumentParser(description="SmartSim Ray head launcher")
parser.add_argument("--port", type=int, help="Port used by Ray to start the Redis server at")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster")
parser.add_argument("--num-cpus", type=int, help="Number of CPUs to be used by Ray head node worker")
args = parser.parse_args()

def current_ip():
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    return socket.gethostbyname(hostname)

cliargs = ["ray",
            "start",
            "--head",
            "--block",
            f"--port={args.port}",
            f"--redis-password={args.redis_password}",
            f"--node-ip-address={current_ip()}"]
if args.num_cpus:
    cliargs.append(f"--num-cpus={args.num_cpus}")

print(" ".join(cliargs))

p = Popen(cliargs,
          stdout=PIPE,
          stderr=STDOUT)

for line in iter(p.stdout.readline, b''):
    print(line.decode('utf-8').rstrip(), flush=True)