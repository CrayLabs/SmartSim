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
import signal
import psutil
import argparse
import tempfile
import filelock
import socket

from typing import List
from pathlib import Path
from subprocess import PIPE, STDOUT

from smartsim._core.utils.network import current_ip
from smartsim.error import SSInternalError
from smartsim.log import get_logger
logger = get_logger(__name__)


def main(network_interface: str, db_cpus: int, command: List[str]):

    try:
        ip_address = current_ip(network_interface)
    except ValueError as e:
        logger.warning(e)
        ip_address = None

    lo_address = current_ip("lo")

    if lo_address == ip_address or not ip_address:
        cmd = command + [f"--bind {lo_address}"]
    else:
        # bind to both addresses if the user specified a network
        # address that exists and is not the loopback address
        cmd = command + [f"--bind {lo_address} {ip_address}"]

    # we generally want to catch all exceptions here as
    # if this process dies, the application will most likely fail
    try:
        p = psutil.Popen(cmd, stdout=PIPE, stderr=STDOUT)
    except Exception as e:
        raise SSInternalError("Co-located process failed to start") from e

    # Set CPU affinity to the last $db_cpus CPUs
    affinity = p.cpu_affinity()
    cpus_to_use = affinity[-db_cpus:]
    p.cpu_affinity(cpus_to_use)

    logger.debug("\n\nCo-located database information\n" +  "\n".join((
        f"\tIP Address: {ip_address}",
        f"\t# of Database CPUs: {db_cpus}",
        f"\tAffinity: {cpus_to_use}",
        f"\tCommand: {' '.join(cmd)}\n\n"
    )))

    for line in iter(p.stdout.readline, b""):
        print(line.decode("utf-8").rstrip(), flush=True)


def cleanup(signo, frame):
    try:
        if LOCK.is_locked:
            LOCK.release()

        if os.path.exists(LOCK.lock_file):
            os.remove(LOCK.lock_file)
    except OSError as e:
        logger.warning(
            f"Failed to clean up co-located database gracefully: {str(e)}"
        )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prefix_chars="+", description="SmartSim Process Launcher"
        )
        parser.add_argument("+ifname", type=str, help="Network Interface name", default="lo")
        parser.add_argument("+lockfile", type=str, help="Filename to create for single proc per host")
        parser.add_argument("+db_cpus", type=int, default=2, help="Number of CPUs to use for DB")
        parser.add_argument("+command", nargs="+", help="Command to run")
        args = parser.parse_args()

        tmp_lockfile = Path(tempfile.gettempdir()) / args.lockfile

        LOCK = filelock.FileLock(tmp_lockfile)
        LOCK.acquire(timeout=0.1)
        logger.debug(f"Starting co-located database on host: {socket.gethostname()}")

        os.environ["PYTHONUNBUFFERED"] = "1"

        # make sure to register the cleanup before the start
        # the proecss so our signaller will be able to stop
        # the database process.
        signal.signal(signal.SIGTERM, cleanup)
        main(args.ifname, args.db_cpus, args.command)

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except filelock.Timeout:
        exit(0)
