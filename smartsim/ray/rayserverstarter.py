import argparse
from subprocess import run, Popen, PIPE, STDOUT
from time import sleep
from smartsim.launcher.util.shell import execute_cmd
import os

parser = argparse.ArgumentParser(description="SmartSim Ray head launcher")
parser.add_argument("--port", type=int, help="Port used by Ray to start the head node at")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster")
parser.add_argument("--num-cpus", type=int, help="Number of CPUs to be used by Ray workers")
args = parser.parse_args()

def current_ip():
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    return socket.gethostbyname(hostname)

args = ["ray", "start", "--head", f"--port={args.port}",
        f"--redis-password={args.redis_password}", f"--num-cpus={args.num_cpus}",
        f"--node-ip-address={current_ip()}",
        "--block",
       ]
print(" ".join(args))
os.environ["PYTHONUNBUFFERED"] = "1"
p = Popen(args,
          stdout=PIPE,
          stderr=STDOUT)

for line in iter(p.stdout.readline, b''):
    print(line.decode('utf-8').rstrip())

# _, out, err = execute_cmd(args, cwd='.')
# print(out)
# print(err)

while True:
    sleep(60)