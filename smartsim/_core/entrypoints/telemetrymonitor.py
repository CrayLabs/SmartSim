# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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
import asyncio
import logging
import os
import pathlib
import signal
import sys
import typing as t
from types import FrameType

import smartsim._core.config as cfg
from smartsim._core.utils.telemetry.telemetry import (
    TelemetryMonitor,
    TelemetryMonitorArgs,
)
from smartsim.log import DEFAULT_LOG_FORMAT, HostnameFilter

"""Telemetry Monitor entrypoint
Starts a long-running, standalone process that hosts a `TelemetryMonitor`"""


logger = logging.getLogger("TelemetryMonitor")


def register_signal_handlers(
    handle_signal: t.Callable[[int, t.Optional[FrameType]], None]
) -> None:
    """Register a signal handling function for all termination events

    :param handle_signal: the function to execute when a term signal is received
    """
    # NOTE: omitting kill because it is not catchable
    term_signals = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
    for signal_num in term_signals:
        signal.signal(signal_num, handle_signal)


def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments

    :returns: An argument parser ready to accept required telemetry monitor parameters
    """
    arg_parser = argparse.ArgumentParser(description="SmartSim Telemetry Monitor")
    arg_parser.add_argument(
        "-exp_dir",
        type=str,
        help="Experiment root directory",
        required=True,
    )
    arg_parser.add_argument(
        "-frequency",
        type=float,
        help="Frequency of telemetry updates (in seconds))",
        required=True,
    )
    arg_parser.add_argument(
        "-cooldown",
        type=int,
        help="Default lifetime of telemetry monitor (in seconds) before auto-shutdown",
        default=cfg.CONFIG.telemetry_cooldown,
    )
    arg_parser.add_argument(
        "-loglevel",
        type=int,
        help="Logging level",
        default=logging.INFO,
    )
    return arg_parser


def parse_arguments() -> TelemetryMonitorArgs:
    """Parse the command line arguments and return an instance
    of TelemetryMonitorArgs populated with the CLI inputs

    :returns: `TelemetryMonitorArgs` instance populated with command line arguments
    """
    parser = get_parser()
    parsed_args = parser.parse_args()
    return TelemetryMonitorArgs(
        parsed_args.exp_dir,
        parsed_args.frequency,
        parsed_args.cooldown,
        parsed_args.loglevel,
    )


def configure_logger(logger_: logging.Logger, log_level_: int, exp_dir: str) -> None:
    """Configure the telemetry monitor logger to write logs to the
    target output file path passed as an argument to the entrypoint

    :param logger_: logger to configure
    :param log_level_: log level to apply to the python logging system
    :param exp_dir: root path to experiment outputs
    """
    logger_.setLevel(log_level_)
    logger_.propagate = False

    # use a standard subdirectory of the experiment output path for logs
    telemetry_dir = pathlib.Path(exp_dir) / cfg.CONFIG.telemetry_subdir

    # all telemetry monitor logs are written to file in addition to stdout
    log_path = telemetry_dir / "logs/telemetrymonitor.out"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, "a")

    # HostnameFilter is required to enrich log context to use DEFAULT_LOG_FORMAT
    file_handler.addFilter(HostnameFilter())

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)


if __name__ == "__main__":
    """Prepare the telemetry monitor process using command line arguments.

    Sample usage:
    python -m smartsim._core.entrypoints.telemetrymonitor -exp_dir <exp_dir>
          -frequency 30 -cooldown 90 -loglevel INFO
    The experiment id is generated during experiment startup
    and can be found in the manifest.json in <exp_dir>/.smartsim/telemetry
    """
    os.environ["PYTHONUNBUFFERED"] = "1"

    args = parse_arguments()
    configure_logger(logger, args.log_level, args.exp_dir)

    telemetry_monitor = TelemetryMonitor(args)

    # Must register cleanup before the main loop is running
    def cleanup_telemetry_monitor(_signo: int, _frame: t.Optional[FrameType]) -> None:
        """Create an enclosure on `manifest_observer` to avoid global variables"""
        telemetry_monitor.cleanup()

    register_signal_handlers(cleanup_telemetry_monitor)

    try:
        asyncio.run(telemetry_monitor.run())
        sys.exit(0)
    except Exception:
        logger.exception(
            "Shutting down telemetry monitor due to unexpected error", exc_info=True
        )

    sys.exit(1)
