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
import json
import os
import signal
import socket
import textwrap
import typing as t
from types import FrameType

import zmq

import smartsim._core.utils.helpers as _helpers
from smartsim._core.launcher.dragon.dragonBackend import DragonBackend
from smartsim._core.schemas import (
    DragonBootstrapRequest,
    DragonBootstrapResponse,
    DragonShutdownResponse,
)
from smartsim._core.schemas.dragonRequests import request_serializer
from smartsim._core.schemas.dragonResponses import response_serializer
from smartsim._core.utils.network import get_best_interface_and_address

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]

SHUTDOWN_INITIATED = False


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    if not signo:
        print("Received signal with no signo")
    else:
        print(f"Received {signo}")
    cleanup()


context = zmq.Context()

"""
Dragon server entrypoint script
"""

DBPID: t.Optional[int] = None


def print_summary(network_interface: str, ip_address: str) -> None:
    zmq_config = {"interface": network_interface, "address": ip_address}

    with open("dragon_config.log", "w", encoding="utf-8") as dragon_config_log:
        dragon_config_log.write(
            textwrap.dedent(f"""\
                -------- Dragon Configuration --------
                IPADDRESS: {ip_address}
                NETWORK: {network_interface}
                HOSTNAME: {socket.gethostname()}
                DRAGON_SERVER_CONFIG: {json.dumps(zmq_config)}
                --------------------------------------

                --------------- Output ---------------

                """),
        )


def run(dragon_head_address: str) -> None:
    global SHUTDOWN_INITIATED  # pylint: disable=global-statement
    print(f"Opening socket {dragon_head_address}")
    dragon_head_socket = context.socket(zmq.REP)
    dragon_head_socket.bind(dragon_head_address)
    dragon_backend = DragonBackend()

    while not SHUTDOWN_INITIATED:
        print(f"Listening to {dragon_head_address}")
        req = dragon_head_socket.recv_json()
        print(f"Received request: {req}")
        drg_req = request_serializer.deserialize_from_json(str(req))
        resp = dragon_backend.process_request(drg_req)
        print(f"Sending response {resp}", flush=True)
        dragon_head_socket.send_json(response_serializer.serialize_to_json(resp))
        if isinstance(resp, DragonShutdownResponse):
            SHUTDOWN_INITIATED = True


def main(args: argparse.Namespace) -> int:
    interface, address = get_best_interface_and_address()
    if not interface:
        raise ValueError("Net interface could not be determined")
    dragon_head_address = f"tcp://{address}"

    if args.launching_address:
        if str(args.launching_address).split(":", maxsplit=1)[0] == dragon_head_address:
            address = "localhost"
            dragon_head_address = "tcp://localhost:5555"
        else:
            dragon_head_address += ":5555"

        launcher_socket = context.socket(zmq.REQ)
        launcher_socket.connect(args.launching_address)

        response = (
            _helpers.start_with(DragonBootstrapRequest(address=dragon_head_address))
            .then(request_serializer.serialize_to_json)
            .then(launcher_socket.send_json)
            .then(lambda _: launcher_socket.recv_json())
            .then(str)
            .then(response_serializer.deserialize_from_json)
            .get_result()
        )

        if not isinstance(response, DragonBootstrapResponse):
            raise ValueError(
                "Could not receive connection confirmation from launcher. Aborting."
            )

    print_summary(interface, dragon_head_address)

    run(dragon_head_address=dragon_head_address)

    print("Shutting down! Bye bye!")
    return 0


def cleanup() -> None:
    global SHUTDOWN_INITIATED  # pylint: disable=global-statement
    print("Cleaning up", flush=True)
    SHUTDOWN_INITIATED = True


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Dragon Head Process"
    )
    parser.add_argument(
        "+launching_address",
        type=str,
        help="Address of launching process if a ZMQ connection can be established",
        required=False,
    )
    parser.add_argument(
        "+interface", type=str, help="Network Interface name", required=False
    )
    args_ = parser.parse_args()

    # make sure to register the cleanup before the start
    # the process so our signaller will be able to stop
    # the database process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)

    raise SystemExit(main(args_))
