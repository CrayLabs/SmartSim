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

import os
import argparse
from typing import List
from subprocess import PIPE, STDOUT, Popen

from smartsim._core.utils.network import current_ip
from smartsim.exp.ray import parse_ray_head_node_address

def main(network_interface: str,
         port: int,
         is_head: bool,
         password: str,
         ray_exe: str,
         ray_args: List[str],
         dash_port: str,
         head_log: str,
         ):

    ip_address = current_ip(network_interface)

    cliargs = [
        ray_exe,
        "start",
        "--head"
        if is_head
        else f"--address={parse_ray_head_node_address(head_log)}:{port}",
        "--block",
        f"--node-ip-address={ip_address}",
    ]

    if ray_args:
        cliargs += ray_args
        if is_head and not any(
            [arg.startswith("--dashboard-host") for arg in ray_args]
        ):
            cliargs += [f"--dashboard-host={ip_address}"]

    if password:
        cliargs += [f"--redis-password={password}"]

    if is_head:
        cliargs += [f"--port={port}", f"--dashboard-port={dash_port}"]


    cmd = " ".join(cliargs)
    print(f"Ray Command: {cmd}")

    p = Popen(cliargs, stdout=PIPE, stderr=STDOUT)

    for line in iter(p.stdout.readline, b""):
        print(line.decode("utf-8").rstrip(), flush=True)


if __name__ == "__main__":

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

    main(args.ifname,
         args.port,
         args.head,
         args.redis_password,
         args.ray_exe,
         args.ray_args,
         args.dashboard_port,
         args.head_log)