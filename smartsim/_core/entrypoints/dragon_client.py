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
import sys
import time
import typing as t

import psutil

from smartsim._core.launcher.dragon.dragonConnector import DragonConnector
from smartsim._core.schemas import (
    DragonRequest,
    DragonShutdownRequest,
    request_registry,
)
from smartsim.log import get_logger

SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]

logger = get_logger("Dragon Client")


def cleanup() -> None:
    logger.debug("Cleaning up")


def main(args: argparse.Namespace) -> int:

    requests: t.List[DragonRequest] = []

    with open(args.submit, "r", encoding="utf-8") as request_file:
        req_strings = json.load(fp=request_file)

    for req_str in req_strings:
        requests.append(request_registry.from_string(req_str))

    requests.append(DragonShutdownRequest(immediate=False))

    connector = DragonConnector()

    for request in requests:
        response = connector.send_request(request)
        if response.error_message is not None:
            logger.error(response.error_message)

    logger.info("Terminated sending requests, waiting for Dragon Server to complete")

    # pylint: disable-next=protected-access
    if connector._dragon_head_pid is None:
        logger.error(
            "Could not get Dragon Server PID and will not be able to monitor it."
        )
        return 1

    while True:
        # pylint: disable-next=protected-access
        if psutil.pid_exists(connector._dragon_head_pid):
            time.sleep(1)
        else:
            break

    return 0


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Dragon server started")

    parser = argparse.ArgumentParser(
        prefix_chars="+",
        description="SmartSim Dragon Client Process, to be used in batch scripts",
    )
    parser.add_argument("+submit", type=str, help="Path to request file", required=True)
    args_ = parser.parse_args()

    sys.exit(main(args_))
