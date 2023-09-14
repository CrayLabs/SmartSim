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
import logging
import os
import pathlib
import psutil
import shlex
import signal
import sys
import typing as t

from datetime import datetime
from subprocess import PIPE  #, STDOUT
from types import FrameType

from smartsim.log import get_logger
from smartsim.telemetrymanager import track_event, PersistableEntity


STEP_PID = None
logger: t.Optional[logging.Logger] = get_logger(__name__)

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
        logger.warning("Received signal with no signo")
    cleanup()


def get_ts() -> int:
    """Helper function to ensure all timestamps are converted to integers"""
    return int(datetime.timestamp(datetime.now()))


def main(
    cmd: str,
    etype: str,
    step_name: str,
    exp_dir: str = "",
) -> int:
    """Execute the step command and emit tracking events"""
    global STEP_PID  # pylint: disable=global-statement

    exp_path = pathlib.Path(exp_dir)
    if not exp_path.exists():
        raise ValueError(f"The experiment directory does not exist: {exp_dir}")

    try:
        logger.debug(f"persisting step start for name: {step_name}, etype: {etype}")

        cleaned_cmd = shlex.split(cmd)
        process = psutil.Popen(cleaned_cmd, stdout=sys.stdout, stderr=sys.stderr)
        STEP_PID = process.pid
        job_id = "" # unmanaged jobs have no job ID, only step ID (the pid)

        persistable = PersistableEntity(
            etype,
            step_name,
            job_id,
            str(STEP_PID),
            get_ts(),
            exp_dir,
        )
        
        track_event(persistable.timestamp, persistable, "start", exp_path, logger)
        rc = process.wait()

        logger.debug(f"persisting step end for name: {step_name} w/return code: {rc}")
        track_event(persistable.timestamp, persistable, "stop", exp_path, logger)
        return rc

    except Exception as e:
        logger.error(f"Failed to execute step: {e}")
    
    return 1


def cleanup() -> None:
    try:
        logger.debug("Cleaning up step executor")
        # attempt to stop the subprocess performing step-execution
        step_proc = psutil.Process(STEP_PID)
        step_proc.terminate()

    except psutil.NoSuchProcess:
        logger.warning("Unable to find step executor process to kill.")

    except OSError as e:
        logger.warning(f"Failed to clean up step executor gracefully: {e}")


def register_signal_handlers() -> None:
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


if __name__ == "__main__":
    # NOTE: plus prefix avoids passing param incorrectly to python interpreter, 
    # e.g. python receives params - `python -m smartsim._core.entrypoints.indirect -c ... -t ...`
    arg_parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Step Executor"
    )
    arg_parser.add_argument(
        "+c", type=str, help="The command to execute", required=True
    )
    arg_parser.add_argument(
        "+t", type=str, help="The type of entity related to the step"
    )
    arg_parser.add_argument(
        "+n", type=str, help="The step name being executed"
    )
    arg_parser.add_argument(
        "+d", type=str, help="The experiment root directory"
    )

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
            exp_dir=parsed_args.d,
        )
        sys.exit(rc)

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except Exception as e:
        logger.exception(f"An unexpected error caused step execution to fail: {e}")
        sys.exit(1)
