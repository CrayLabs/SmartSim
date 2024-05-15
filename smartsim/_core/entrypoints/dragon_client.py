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
import sys
import time
import typing as t
from pathlib import Path
from types import FrameType

import zmq

from smartsim._core.launcher.dragon.dragonConnector import DragonConnector
from smartsim._core.schemas import (
    DragonHandshakeRequest,
    DragonRequest,
    DragonShutdownRequest,
    request_registry,
)
from smartsim.log import get_logger

"""
Dragon client entrypoint script, used to start a server, send requests to it
and then shut it down.
"""

logger = get_logger("Dragon Client")

SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]


@dataclasses.dataclass
class DragonClientEntrypointArgs:
    submit: Path


def cleanup() -> None:
    """Cleanup resources"""
    logger.debug("Cleaning up")


def parse_requests(request_filepath: Path) -> t.List[DragonRequest]:
    """Parse serialized requests from file

    :param request_filepath: Path to file with serialized requests
    :return: Deserialized requests
    """
    requests: t.List[DragonRequest] = []
    try:
        with open(request_filepath, "r", encoding="utf-8") as request_file:
            req_strings = json.load(fp=request_file)
    except FileNotFoundError as e:
        logger.error(
            "Could not find file with run requests,"
            f"please check whether {request_filepath} exists."
        )
        raise e from None
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode request file {request_filepath}.")
        raise e from None

    requests = [request_registry.from_string(req_str) for req_str in req_strings]

    return requests


def parse_arguments(args: t.List[str]) -> DragonClientEntrypointArgs:
    """Parse arguments used to run entrypoint script

    :param args: Arguments without name of executable
    :raises ValueError: If the request file is not specified
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prefix_chars="+",
        description="SmartSim Dragon Client Process, to be used in batch scripts",
    )
    parser.add_argument("+submit", type=str, help="Path to request file", required=True)
    args_ = parser.parse_args(args)

    if not args_.submit:
        raise ValueError("Request file not provided.")

    return DragonClientEntrypointArgs(submit=Path(args_.submit))


def handle_signal(signo: int, _frame: t.Optional[FrameType] = None) -> None:
    """Handle signals sent to this process

    :param signo: Signal number
    :param _frame: Frame, defaults to None
    """
    if not signo:
        logger.info("Received signal with no signo")
    else:
        logger.info(f"Received signal {signo}")
    cleanup()


def register_signal_handlers() -> None:
    """Register signal handlers prior to execution"""
    # make sure to register the cleanup before the start
    # the process so our signaller will be able to stop
    # the server process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def execute_entrypoint(args: DragonClientEntrypointArgs) -> int:
    """Execute the entrypoint with specified arguments

    :param args: Parsed arguments
    :return: Return code
    """

    try:
        requests = parse_requests(args.submit)
    except Exception:
        logger.error("Dragon client failed to parse request file", exc_info=True)
        return os.EX_OSFILE

    requests.append(DragonShutdownRequest(immediate=False, frontend_shutdown=True))

    connector = DragonConnector()

    for request in requests:
        response = connector.send_request(request)
        if response.error_message is not None:
            logger.error(response.error_message)

    logger.info("Terminated sending requests, waiting for Dragon Server to complete")

    if not connector.can_monitor:
        logger.error(
            "Could not get Dragon Server PID and will not be able to monitor it."
        )
        return os.EX_IOERR

    while True:
        try:
            time.sleep(5)
            connector.send_request(DragonHandshakeRequest())
        except zmq.error.Again:
            logger.debug("Could not reach server, assuming backend has shut down")
            break

    logger.info("Client has finished.")

    return os.EX_OK


def main(args_: t.List[str]) -> int:
    """Execute the dragon client entrypoint as a module"""

    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Dragon client started")

    args = parse_arguments(args_)
    register_signal_handlers()

    try:
        return execute_entrypoint(args)
    except Exception:
        logger.error(
            "An unexpected error occurred in the Dragon client entrypoint",
            exc_info=True,
        )
    finally:
        cleanup()

    return os.EX_SOFTWARE


if __name__ == "__main__":

    sys.exit(main(sys.argv[1:]))
