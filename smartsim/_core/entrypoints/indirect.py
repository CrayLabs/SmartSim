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
import pathlib
import psutil
import signal
import sys
import time
import typing as t

from types import FrameType

from smartsim.log import get_logger
from smartsim._core.entrypoints.telemetrymonitor import track_event
from smartsim._core.utils.helpers import get_ts, decode_cmd


STEP_PID: t.Optional[int] = None
logger = get_logger(__name__)

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]


def main(
    cmd: str,
    etype: str,
    output_path: str,
    error_path: str,
    cwd: str,
    status_dir: str,
) -> int:
    """Execute the step command and emit tracking events"""
    global STEP_PID  # pylint: disable=global-statement
    proxy_pid = os.getpid()

    status_path = pathlib.Path(status_dir)
    if not status_path.exists():
        status_path.mkdir(parents=True, exist_ok=True)

    if not cmd.strip():
        raise ValueError("Invalid cmd supplied")

    cleaned_cmd = decode_cmd(cmd)
    ret_code: int = 1
    logger.debug("Indirect step starting")

    with open(output_path, "w+", encoding="utf-8") as ofp, open(
        error_path, "w+", encoding="utf-8"
    ) as efp:
        start_detail = f"Proxy process {proxy_pid}"
        start_rc: t.Optional[int] = None

        try:
            process = psutil.Popen(
                cleaned_cmd,
                cwd=cwd,
                stdout=ofp.fileno(),
                stderr=efp.fileno(),
                close_fds=True,
            )
            STEP_PID = process.pid
            logger.info(f"Indirect step step {STEP_PID} started")
            start_detail += f" started child process {STEP_PID}"

        except Exception as ex:
            start_detail += f" failed to start child process. {ex}"
            start_rc = 1
            logger.error("Failed to create process", exc_info=True)
            cleanup()
            return 1
        finally:
            track_event(
                get_ts(),
                proxy_pid,
                "", # step_id for unmanaged task is always empty
                etype,
                "start",
                status_path,
                logger,
                detail=start_detail,
                return_code=start_rc
            )

        ret_code = process.wait()

    logger.info(f"Indirect step {STEP_PID} complete")
    msg = f"Process {STEP_PID} finished with return code: {ret_code}"
    track_event(
        get_ts(),
        proxy_pid,
        "", # step_id for unmanaged task is always empty
        etype,
        "stop",
        status_path,
        logger,
        detail=msg,
        return_code=ret_code,
    )
    cleanup()

    return ret_code


def cleanup() -> None:
    """Perform cleanup required for clean termination"""
    global STEP_PID  # pylint: disable=global-statement
    if STEP_PID is not None and STEP_PID < 1:
        return

    try:
        # attempt to stop the subprocess performing step-execution
        if STEP_PID is not None and psutil.pid_exists(STEP_PID):
            process = psutil.Process(STEP_PID)
            process.terminate()

    except psutil.NoSuchProcess:
        # swallow exception to avoid overwriting outputs from cmd
        ...

    except OSError as ex:
        logger.warning(f"Failed to clean up step executor gracefully: {ex}")
    finally:
        STEP_PID = 0


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
        logger.warning("Received signal with no signo")

    cleanup()


def register_signal_handlers() -> None:
    """Register a signal handling function for all termination events"""
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def get_parser() -> argparse.ArgumentParser:
    # NOTE: plus prefix avoids passing param incorrectly to python interpreter,
    # e.g. `python -m smartsim._core.entrypoints.indirect -c ... -t ...`
    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Step Executor"
    )
    parser.add_argument("+c", type=str, help="The command to execute", required=True)
    parser.add_argument(
        "+t", type=str, help="The type of entity related to the step", required=True
    )
    parser.add_argument(
        "+w", type=str, help="The working directory of the executable", required=True
    )
    parser.add_argument("+o", type=str, help="Output file", required=True)
    parser.add_argument("+e", type=str, help="Erorr output file", required=True)
    parser.add_argument(
        "+d", type=str, help="Directory for telemetry output", required=True
    )
    return parser


if __name__ == "__main__":
    arg_parser = get_parser()
    os.environ["PYTHONUNBUFFERED"] = "1"

    try:
        parsed_args = arg_parser.parse_args()
        logger.debug("Starting indirect step execution")

        # make sure to register the cleanup before the start the process
        # so our signaller will be able to stop the database process.
        register_signal_handlers()

        rc = main(
            cmd=parsed_args.c,
            etype=parsed_args.t,
            output_path=parsed_args.o,
            error_path=parsed_args.e,
            cwd=parsed_args.w,
            status_dir=parsed_args.d,
        )
        sys.exit(rc)

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except Exception as e:
        logger.exception(f"An unexpected error caused step execution to fail: {e}")
        sys.exit(1)
