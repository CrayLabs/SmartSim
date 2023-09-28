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
import base64
import os
import pathlib
import psutil
import signal
import sys
import time
import typing as t

from datetime import datetime
from types import FrameType

from smartsim.log import get_logger
from smartsim._core.entrypoints.telemetrymonitor import track_event


STEP_PID = None
logger = get_logger(__name__)

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]


def get_ts() -> int:
    """Helper function to ensure all timestamps are converted to integers"""
    return int(datetime.timestamp(datetime.now()))


def main(
    cmd: str,
    etype: str,
    step_name: str,
    output_path: str,
    error_path: str,
    exp_dir: str = "",
) -> int:
    """Execute the step command and emit tracking events"""
    global STEP_PID  # pylint: disable=global-statement

    exp_path = pathlib.Path(exp_dir)
    if not exp_path.exists():
        raise ValueError(f"The experiment directory does not exist: {exp_dir}")

    if not cmd.strip():
        raise ValueError("Invalid cmd supplied")

    decoded_cmd = base64.b64decode(cmd.encode("ascii"))
    cleaned_cmd = decoded_cmd.decode("ascii").split("|")
    # if not cleaned_cmd:
    #     raise ValueError(f"Invalid cmd supplied: {cmd}")

    job_id = ""  # unmanaged jobs have no job ID, only step ID (the pid)

    try:
        ofp = open(  # pylint: disable=consider-using-with
            output_path, "w+", encoding="utf-8"
        )
        efp = open(  # pylint: disable=consider-using-with
            error_path, "w+", encoding="utf-8"
        )

        process = psutil.Popen(
            cleaned_cmd,
            cwd=exp_dir,
            stdout=ofp.fileno(),
            stderr=efp.fileno(),
            close_fds=True,
        )
        STEP_PID = process.pid

        track_event(
            get_ts(), step_name, job_id, str(STEP_PID), etype, "start", exp_path, logger
        )

    except Exception:
        logger.error("Failed to create process", exc_info=True)
        cleanup()
        return 1

    ret_code: t.Optional[int] = None

    try:
        while all((process.is_running(), ret_code is None, STEP_PID > 0)):
            ret_code = process.poll()
            if ret_code is not None:
                break
            time.sleep(1)

        track_event(
            get_ts(),
            step_name,
            job_id,
            str(STEP_PID),
            etype,
            "stop",
            exp_path,
            logger,
            detail=f"process {STEP_PID} finished with return code: {ret_code}",
            return_code=ret_code,
        )
        return ret_code
    except Exception:
        logger.error("Failed to execute process", exc_info=True)
    finally:
        cleanup()

    return 1


def cleanup() -> None:
    """Perform cleanup required for clean termination"""
    global STEP_PID  # pylint: disable=global-statement
    if STEP_PID < 1:
        return

    try:
        # attempt to stop the subprocess performing step-execution
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
        "+n", type=str, help="The step name being executed", required=True
    )
    parser.add_argument(
        "+d", type=str, help="The experiment root directory", required=True
    )
    parser.add_argument("+o", type=str, help="Output file", required=True)
    parser.add_argument("+e", type=str, help="Erorr output file", required=True)
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
            step_name=parsed_args.n,
            # cwd=parsed_args.w,
            output_path=parsed_args.o,
            error_path=parsed_args.e,
            exp_dir=parsed_args.d,
        )
        sys.exit(rc)

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except Exception as e:
        logger.exception(f"An unexpected error caused step execution to fail: {e}")
        sys.exit(1)
