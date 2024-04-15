# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterpris
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
import sys
import textwrap
import typing as t
from types import FrameType

import zmq
import zmq.auth.thread

from smartsim._core.launcher.dragon import dragonSockets
from smartsim._core.launcher.dragon.dragonBackend import DragonBackend
from smartsim._core.schemas import DragonBootstrapRequest, DragonBootstrapResponse
from smartsim._core.utils.network import get_best_interface_and_address
from smartsim.log import get_logger

logger = get_logger("Dragon Server")

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]

SHUTDOWN_INITIATED = False


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    if not signo:
        logger.info("Received signal with no signo")
    else:
        logger.info(f"Received signal {signo}")
    cleanup()


"""
Dragon server entrypoint script
"""


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


def run(
    zmq_context: "zmq.Context[t.Any]",
    dragon_head_address: str,
    dragon_pid: int,
) -> None:
    logger.debug(f"Opening socket {dragon_head_address}")

    zmq_context.setsockopt(zmq.SNDTIMEO, value=1000)
    zmq_context.setsockopt(zmq.RCVTIMEO, value=1000)
    zmq_context.setsockopt(zmq.REQ_CORRELATE, 1)
    zmq_context.setsockopt(zmq.REQ_RELAXED, 1)

    dragon_head_socket = dragonSockets.get_secure_socket(zmq_context, zmq.REP, True)
    dragon_head_socket.bind(dragon_head_address)
    dragon_backend = DragonBackend(pid=dragon_pid)

    server = dragonSockets.as_server(dragon_head_socket)

    logger.debug(f"Listening to {dragon_head_address}")
    while not (dragon_backend.should_shutdown or SHUTDOWN_INITIATED):
        try:
            req = server.recv()
            logger.debug(f"Received {type(req).__name__} {req}")
        except zmq.Again:
            # dragon_backend.print_status()
            dragon_backend.update()
            continue

        resp = dragon_backend.process_request(req)

        logger.debug(f"Sending {type(resp).__name__} {resp}")
        try:
            server.send(resp)
        except zmq.Again:
            logger.error("Could not send response back to launcher.")

        dragon_backend.print_status()
        dragon_backend.update()
        if not (dragon_backend.should_shutdown or SHUTDOWN_INITIATED):
            logger.debug(f"Listening to {dragon_head_address}")
        else:
            logger.info("Shutdown has been requested")
            break


def main(args: argparse.Namespace) -> int:
    if_config = get_best_interface_and_address()
    interface = if_config.interface
    address = if_config.address
    if not interface:
        raise ValueError("Net interface could not be determined")
    dragon_head_address = f"tcp://{address}"

    if args.launching_address:
        zmq_context = zmq.Context()

        if str(args.launching_address).split(":", maxsplit=1)[0] == dragon_head_address:
            address = "localhost"
            dragon_head_address = "tcp://localhost:5555"
        else:
            dragon_head_address += ":5555"

        zmq_authenticator = dragonSockets.get_authenticator(zmq_context)

        logger.debug("Getting launcher socket")
        launcher_socket = dragonSockets.get_secure_socket(zmq_context, zmq.REQ, False)

        logger.debug(f"Connecting launcher socket to: {args.launching_address}")
        launcher_socket.connect(args.launching_address)
        client = dragonSockets.as_client(launcher_socket)

        logger.debug(
            f"Sending bootstrap request to launcher_socket with {dragon_head_address}"
        )
        client.send(DragonBootstrapRequest(address=dragon_head_address))
        response = client.recv()

        logger.debug(f"Received bootstrap response: {response}")
        if not isinstance(response, DragonBootstrapResponse):
            raise ValueError(
                "Could not receive connection confirmation from launcher. Aborting."
            )

        print_summary(interface, dragon_head_address)

        try:
            logger.debug("Executing event loop")
            run(
                zmq_context=zmq_context,
                dragon_head_address=dragon_head_address,
                dragon_pid=response.dragon_pid,
            )
        except Exception as e:
            logger.error(f"Dragon server failed with {e}", exc_info=True)
            return os.EX_SOFTWARE
        finally:
            if zmq_authenticator is not None and zmq_authenticator.is_alive():
                zmq_authenticator.stop()

    logger.info("Shutting down! Bye bye!")
    return 0


def cleanup() -> None:
    global SHUTDOWN_INITIATED  # pylint: disable=global-statement
    logger.debug("Cleaning up")
    SHUTDOWN_INITIATED = True


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Dragon server started")

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

    sys.exit(main(args_))
