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
import dataclasses
import json
import os
import signal
import socket
import sys
import textwrap
import time
import typing as t
from types import FrameType

import zmq
import zmq.auth.thread

from smartsim._core.config import get_config
from smartsim._core.launcher.dragon import dragon_sockets
from smartsim._core.launcher.dragon.dragon_backend import DragonBackend
from smartsim._core.schemas import (
    DragonBootstrapRequest,
    DragonBootstrapResponse,
    DragonShutdownRequest,
)
from smartsim._core.utils.network import get_best_interface_and_address
from smartsim.log import ContextThread, get_logger

"""
Dragon server entrypoint script
"""

logger = get_logger("Dragon Server")

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]

SHUTDOWN_INITIATED = False


@dataclasses.dataclass
class DragonEntrypointArgs:
    launching_address: str
    interface: str


def handle_signal(signo: int, _frame: t.Optional[FrameType] = None) -> None:
    if not signo:
        logger.info("Received signal with no signo")
    else:
        logger.info(f"Received signal {signo}")
    cleanup()


def get_log_path() -> str:
    config = get_config()
    return config.dragon_log_filename


def print_summary(network_interface: str, ip_address: str) -> None:
    zmq_config = {"interface": network_interface, "address": ip_address}

    log_path = get_log_path()
    with open(log_path, "w", encoding="utf-8") as dragon_config_log:
        dragon_config_log.write(
            textwrap.dedent(f"""\
                -------- Dragon Configuration --------
                IPADDRESS: {ip_address}
                NETWORK: {network_interface}
                HOSTNAME: {socket.gethostname()}
                DRAGON_SERVER_CONFIG: {json.dumps(zmq_config)}
                --------------------------------------
                """),
        )


def start_updater(
    backend: DragonBackend, updater: t.Optional[ContextThread]
) -> ContextThread:
    """Start the ``DragonBackend`` updater thread.

    If ``updater`` is not None, then it is first checked and if it
    alive, no other thread is started.

    :param backend: The dragon backend for which the thread will be started
    :param updater: An existing updater thread that might have to be replaced
    :return: Running updater thread
    """
    # If the updater was started, check if it completed or died
    if updater is not None:
        updater.join(0.1)
        # If it's alive, there is nothing to do
        if updater.is_alive():
            return updater
    updater = ContextThread(name="DragonBackend", daemon=True, target=backend.update)
    updater.start()
    return updater


def is_updater_healthy(backend: DragonBackend) -> bool:
    """Check if the backend has been updated recently.

    The acceptable delay is defined as the server timeout plus the backend's cooldown
    period. If the server timeout is set to `-1`, then the acceptable delay is set to
    one minute plus the cooldown period.

    :param backend: The backend for which the updater's health is checked
    :return: Whether the backend was updated recently
    """
    server_timeout = get_config().dragon_server_timeout / 1000
    acceptable_delay = backend.cooldown_period + (
        60.0 if server_timeout == -1 else server_timeout
    )

    heartbeat_delay = backend.current_time - backend.last_heartbeat
    if heartbeat_delay > acceptable_delay:
        logger.debug(
            f"Updater inactive for {heartbeat_delay:.2f} seconds, will request restart."
        )
        return False
    return True


def updater_fallback(backend: DragonBackend, updater: ContextThread) -> ContextThread:
    """Check if updater has updated the backend recently, if not, check its status
    and start a new one if it is not alive.
    :param backend: The dragon backend for which the udpater's health must be checked
    :param updater: The updater thread which has to be checked and (possibly) replaced
    :return: Running updater thread
    """
    if is_updater_healthy(backend):
        return updater
    return start_updater(backend, updater)


# pylint: disable-next=too-many-statements
def run(
    zmq_context: "zmq.Context[t.Any]",
    dragon_head_address: str,
    dragon_pid: int,
) -> None:
    logger.debug(f"Opening socket {dragon_head_address}")
    dragon_head_socket = dragon_sockets.get_secure_socket(zmq_context, zmq.REP, True)
    dragon_head_socket.bind(dragon_head_address)
    dragon_backend = DragonBackend(pid=dragon_pid)

    backend_updater = start_updater(dragon_backend, None)
    server = dragon_sockets.as_server(dragon_head_socket)

    logger.debug(f"Listening to {dragon_head_address}")

    while not dragon_backend.should_shutdown:
        try:
            req = server.recv()
            logger.debug(f"Received {type(req).__name__} {req}")
        except zmq.Again:
            backend_updater = updater_fallback(dragon_backend, backend_updater)
            continue

        resp = dragon_backend.process_request(req)

        logger.debug(f"Sending {type(resp).__name__} {resp}")
        try:
            server.send(resp)
        except zmq.Again:
            logger.error("Could not send response back to launcher.")
            backend_updater = updater_fallback(dragon_backend, backend_updater)

        # We can only check the heartbeat if the backend has not shut down
        if not dragon_backend.should_shutdown:
            logger.debug(f"Listening to {dragon_head_address}")
            backend_updater = updater_fallback(dragon_backend, backend_updater)

        if SHUTDOWN_INITIATED:
            dragon_backend.process_request(DragonShutdownRequest())

    logger.info("Backend shutdown has been requested")

    if backend_updater.is_alive():
        backend_updater.join(1)

    if not dragon_backend.frontend_shutdown:
        logger.info("Frontend will have to be shut down externally")
        while True:
            logger.info("Waiting for external shutdown")
            time.sleep(5)


def execute_entrypoint(args: DragonEntrypointArgs) -> int:
    if_config = get_best_interface_and_address()
    interface = if_config.interface
    address = if_config.address
    if not interface:
        raise ValueError("Net interface could not be determined")
    dragon_head_address = f"tcp://{address}"

    smartsim_config = get_config()
    if args.launching_address:
        zmq_context = zmq.Context()
        zmq_context.setsockopt(
            zmq.SNDTIMEO, value=smartsim_config.dragon_server_timeout
        )
        zmq_context.setsockopt(
            zmq.RCVTIMEO, value=smartsim_config.dragon_server_timeout
        )
        zmq_context.setsockopt(zmq.REQ_CORRELATE, 1)
        zmq_context.setsockopt(zmq.REQ_RELAXED, 1)

        if str(args.launching_address).split(":", maxsplit=1)[0] == dragon_head_address:
            address = "localhost"
            dragon_head_address = "tcp://localhost:5555"
        else:
            dragon_head_address += ":5555"

        zmq_authenticator = dragon_sockets.get_authenticator(zmq_context, timeout=-1)

        logger.debug("Getting launcher socket")
        launcher_socket = dragon_sockets.get_secure_socket(zmq_context, zmq.REQ, False)

        logger.debug(f"Connecting launcher socket to: {args.launching_address}")
        launcher_socket.connect(args.launching_address)
        client = dragon_sockets.as_client(launcher_socket)

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


def remove_config_log() -> None:
    """Remove the Dragon `config_log` file from the file system. Used to
    clean up after a dragon environment is shutdown to eliminate an
    unnecessary attempt to connect to a stopped ZMQ server."""
    log_path = get_log_path()
    if os.path.exists(log_path):
        os.remove(log_path)


def cleanup() -> None:
    global SHUTDOWN_INITIATED  # pylint: disable=global-statement
    logger.debug("Cleaning up")
    remove_config_log()
    SHUTDOWN_INITIATED = True


def register_signal_handlers() -> None:
    # make sure to register the cleanup before the start
    # the process so our signaller will be able to stop
    # the feature store process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def parse_arguments(args: t.List[str]) -> DragonEntrypointArgs:
    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Dragon Head Process"
    )
    parser.add_argument(
        "+launching_address",
        type=str,
        help="Address of launching process if a ZMQ connection can be established",
        required=True,
    )
    parser.add_argument(
        "+interface",
        type=str,
        help="Network Interface name",
        required=False,
    )
    args_ = parser.parse_args(args)

    if not args_.launching_address:
        raise ValueError("Empty launching address supplied.")

    return DragonEntrypointArgs(args_.launching_address, args_.interface)


def main(args_: t.List[str]) -> int:
    """Execute the dragon entrypoint as a module"""
    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Dragon server started")

    args = parse_arguments(args_)
    register_signal_handlers()

    try:
        return_code = execute_entrypoint(args)
        return return_code
    except Exception:
        logger.error(
            "An unexpected error occurred in the Dragon entrypoint.", exc_info=True
        )
    finally:
        cleanup()

    return -1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
