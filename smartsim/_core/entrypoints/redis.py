# BSD 2-Clause License
#
# Copyright (c) 2021-2023 Hewlett Packard Enterprise
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

import argparse
import os
import psutil
import signal
import typing as t

from smartsim._core.utils.network import current_ip
from smartsim.error import SSInternalError
from smartsim.log import get_logger
from subprocess import PIPE, STDOUT
from types import FrameType

logger = get_logger(__name__)

"""
Redis/KeyDB entrypoint script
"""

DBPID = None

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    if not signo:
        logger.warning("Received signal with no signo")
    cleanup()


def build_bind_args(ip_addresses: t.List[str]) -> t.List[str]:
    bind_arg = f"--bind {' '.join(ip_addresses)}"
    # pin source address to avoid random selection by Redis
    bind_src_arg = f"--bind-source-addr {ip_addresses[0]}"
    return [bind_arg, bind_src_arg]


def print_summary(cmd: t.List[str], ip_address: str, network_interface: str) -> None:
    print("-" * 10, "  Running  Command  ", "-" * 10, "\n", flush=True)
    print(f"COMMAND: {' '.join(cmd)}\n", flush=True)
    print(f"IPADDRESS: {ip_address}\n", flush=True)
    print(f"NETWORK: {network_interface}\n", flush=True)
    print("-" * 30, "\n\n", flush=True)
    print("-" * 10, "  Output  ", "-" * 10, "\n\n", flush=True)


def main(network_interface: str, command: t.List[str]) -> None:
    global DBPID  # pylint: disable=global-statement

    try:
        ip_addresses = [current_ip(net_if) for net_if in network_interface.split(",")]
        cmd = command + build_bind_args(ip_addresses)

        print_summary(cmd, ip_addresses[0], network_interface)

        process = psutil.Popen(cmd, stdout=PIPE, stderr=STDOUT)
        DBPID = process.pid

        for line in iter(process.stdout.readline, b""):
            print(line.decode("utf-8").rstrip(), flush=True)
    except Exception as e:
        cleanup()
        raise SSInternalError("Database process starter raised an exception") from e


def cleanup() -> None:
    try:
        logger.debug("Cleaning up database instance")
        # attempt to stop the database process
        db_proc = psutil.Process(DBPID)
        db_proc.terminate()

    except psutil.NoSuchProcess:
        logger.warning("Couldn't find database process to kill.")

    except OSError as e:
        logger.warning(f"Failed to clean up database gracefully: {str(e)}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Process Launcher"
    )
    parser.add_argument(
        "+ifname", type=str, help="Network Interface name", default="lo"
    )
    parser.add_argument("+command", nargs="+", help="Command to run")
    args = parser.parse_args()

    # make sure to register the cleanup before the start
    # the proecss so our signaller will be able to stop
    # the database process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)

    main(args.ifname, args.command)
