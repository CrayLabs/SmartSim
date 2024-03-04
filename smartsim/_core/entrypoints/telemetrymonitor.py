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

from watchdog.events import LoggingEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import get_ts_ms
from smartsim._core.utils.serialize import MANIFEST_FILENAME
from smartsim._core.utils.telemetry.telemetry import ManifestEventHandler
from smartsim.log import DEFAULT_LOG_FORMAT, HostnameFilter

"""Telemetry Monitor entrypoint"""

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]
F_MIN, F_MAX = 1.0, 600.0
_LOG_FILE_NAME = "logs/telemetrymonitor.out"


logger = logging.getLogger("TelemetryMonitor")


def can_shutdown(action_handler: ManifestEventHandler) -> bool:
    managed_jobs = action_handler.job_manager.jobs
    unmanaged_jobs = action_handler.tracked_jobs
    db_jobs = list(filter(lambda j: j.is_db and not j.is_complete, unmanaged_jobs))

    n_jobs, n_dbs = len(managed_jobs), len(db_jobs)
    shutdown_ok = n_jobs + n_dbs == 0

    logger.debug(f"{n_jobs} active job(s), {n_dbs} active db(s)")
    return shutdown_ok


async def event_loop(
    observer: BaseObserver,
    action_handler: ManifestEventHandler,
    frequency: int,
    cooldown_duration: int,
) -> None:
    """Executes all attached timestep handlers every <frequency> milliseconds

    :param observer: (optional) a preconfigured watchdog Observer to inject
    :type observer: t.Optional[BaseObserver]
    :param action_handler: The manifest event processor instance
    :type action_handler: ManifestEventHandler
    :param frequency: frequency (in milliseconds) of update loop
    :type frequency: t.Union[int, float]
    :param logger: a preconfigured Logger instance
    :type logger: logging.Logger
    :param cooldown_duration: number of milliseconds the telemetry monitor should
                              poll for new jobs before attempting to shutdown
    :type cooldown_duration: int
    """
    elapsed: int = 0
    last_ts: int = get_ts_ms()
    shutdown_in_progress = False

    while observer.is_alive() and not shutdown_in_progress:
        duration_ms = 0
        start_ts = get_ts_ms()
        logger.debug(f"Timestep: {start_ts}")
        await action_handler.on_timestep(start_ts)

        elapsed += start_ts - last_ts
        last_ts = start_ts

        if can_shutdown(action_handler):
            if elapsed >= cooldown_duration:
                shutdown_in_progress = True
                logger.info("Beginning telemetry manager shutdown")
                await action_handler.shutdown()
                logger.info("Beginning file monitor shutdown")
                observer.stop()  # type: ignore
                logger.info("Event loop shutdown complete")
                break
        else:
            # reset cooldown any time there are still jobs running
            elapsed = 0

        # track time elapsed to execute metric collection
        duration_ms = get_ts_ms() - start_ts
        wait_ms = max(frequency - duration_ms, 0)

        # delay next loop if collection time didn't exceed loop frequency
        if wait_ms > 0:
            await asyncio.sleep(wait_ms / 1000)  # convert to seconds for sleep

    logger.info("Exiting telemetry monitor event loop")


async def main(
    exp_id: str,
    frequency: t.Union[int, float],
    experiment_dir: pathlib.Path,
    observer: t.Optional[BaseObserver] = None,
    cooldown_duration: t.Optional[int] = 0,
) -> int:
    """Setup the monitoring entities and start the timer-based loop that
    will poll for telemetry data

    :param frequency: frequency (in seconds) of update loop
    :type frequency: t.Union[int, float]
    :param experiment_dir: the experiement directory to monitor for changes
    :type experiment_dir: pathlib.Path
    :param logger: a preconfigured Logger instance
    :type logger: logging.Logger
    :param observer: (optional) a preconfigured Observer to inject
    :type observer: t.Optional[BaseObserver]
    :param cooldown_duration: number of seconds the telemetry monitor should
                              poll for new jobs before attempting to shutdown
    :type cooldown_duration: int
    """
    telemetry_path = experiment_dir / pathlib.Path(CONFIG.telemetry_subdir)
    manifest_path = telemetry_path / MANIFEST_FILENAME

    logger.info(
        f"Executing telemetry monitor - frequency: {frequency}s"
        f", target directory: {experiment_dir}"
        f", telemetry path: {telemetry_path}"
    )

    cooldown_ms = 1000 * (cooldown_duration or CONFIG.telemetry_cooldown)
    log_handler = LoggingEventHandler(logger)  # type: ignore
    frequency_ms = int(frequency * 1000)  # limit collector execution time
    action_handler = ManifestEventHandler(
        exp_id,
        str(MANIFEST_FILENAME),
        timeout_ms=frequency_ms,
        ignore_patterns=["*.out", "*.err"],
    )

    if observer is None:
        observer = Observer()

    try:
        if manifest_path.exists():
            # a manifest may not exist depending on startup timing
            action_handler.process_manifest(str(manifest_path))

        observer.schedule(log_handler, telemetry_path)  # type:ignore
        observer.schedule(action_handler, telemetry_path)  # type:ignore
        observer.start()  # type: ignore

        await event_loop(observer, action_handler, frequency_ms, cooldown_ms)
        return os.EX_OK
    except Exception as ex:
        logger.error(ex)
    finally:
        if observer.is_alive():
            observer.stop()  # type: ignore
            observer.join()
        await action_handler.shutdown()
        logger.debug("Telemetry monitor shutdown complete")

    return os.EX_SOFTWARE


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    """Helper function to ensure clean process termination"""
    if not signo:
        logger.warning("Received signal with no signo")


def register_signal_handlers() -> None:
    """Register a signal handling function for all termination events"""
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments"""
    arg_parser = argparse.ArgumentParser(description="SmartSim Telemetry Monitor")
    arg_parser.add_argument(
        "-frequency",
        type=float,
        help="Frequency of telemetry updates (in seconds))",
        required=True,
    )
    arg_parser.add_argument(
        "-exp_dir",
        type=str,
        help="Experiment root directory",
        required=True,
    )
    arg_parser.add_argument(
        "-cooldown",
        type=int,
        help="Default lifetime of telemetry monitor (in seconds) before auto-shutdown",
        default=CONFIG.telemetry_cooldown,
    )
    arg_parser.add_argument(
        "-loglevel",
        type=int,
        help="Logging level",
        default=logging.DEBUG,
    )
    arg_parser.add_argument(
        "-exp_id",
        type=str,
        help="Unique ID of the parent experiment executing a run",
        required=True,
    )
    return arg_parser


def check_frequency(frequency: t.Union[int, float]) -> None:
    freq_tpl = "Telemetry collection frequency must be {0} {1}s"
    if frequency < F_MIN:
        raise ValueError(freq_tpl.format("greater than", F_MIN))
    if frequency > F_MAX:
        raise ValueError(freq_tpl.format("less than", F_MAX))


def configure_logger(_logger: logging.Logger, arg_ns: argparse.Namespace) -> None:
    log_level: int = (
        arg_ns.loglevel
        if arg_ns.loglevel
        in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        else logging.DEBUG
    )
    _logger.setLevel(log_level)
    _logger.propagate = False

    telem_dir = pathlib.Path(arg_ns.exp_dir) / CONFIG.telemetry_subdir
    log_path = telem_dir / _LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    file_handler = logging.FileHandler(log_path, "a")
    file_handler.addFilter(HostnameFilter())
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = get_parser()
    run_args = parser.parse_args()

    configure_logger(logger, run_args)

    # Must register cleanup before the main loop is running
    register_signal_handlers()
    check_frequency(float(run_args.frequency))

    try:
        asyncio.run(
            main(
                run_args.exp_id,
                int(run_args.frequency),
                pathlib.Path(run_args.exp_dir),
                cooldown_duration=run_args.cooldown,
            )
        )
        sys.exit(0)
    except Exception:
        logger.exception(
            "Shutting down telemetry monitor due to unexpected error", exc_info=True
        )

    sys.exit(1)
