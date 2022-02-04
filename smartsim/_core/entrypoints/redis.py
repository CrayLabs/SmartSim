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

"""
Redis/KeyDB entrypoint script
"""

def main(network_interface: str, command: List[str]):

    ip_address = current_ip(network_interface)
    cmd = command + [f"--bind {ip_address}"]

    print("-" * 10, "  Running  Command  ", "-" * 10, "\n")
    print(f"COMMAND: {' '.join(cmd)}\n")
    print(f"IPADDRESS: {ip_address}\n")
    print(f"NETWORK: {network_interface}\n")
    print("-" * 30, "\n\n")

    print("-" * 10, "  Output  ", "-" * 10, "\n\n")

    p = Popen(cmd, stdout=PIPE, stderr=STDOUT)

    for line in iter(p.stdout.readline, b""):
        print(line.decode("utf-8").rstrip(), flush=True)




if __name__ == "__main__":

    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Process Launcher"
    )
    parser.add_argument("+ifname", type=str, help="Network Interface name", default="lo")
    parser.add_argument("+command", nargs="+", help="Command to run")
    args = parser.parse_args()

    main(args.ifname, args.command)