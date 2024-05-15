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
import asyncio
import json
import logging
import os
import pathlib
import threading
import typing as t

from watchdog.events import (
    FileSystemEvent,
    LoggingEventHandler,
    PatternMatchingEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from smartsim._core.config import CONFIG
from smartsim._core.control.job import JobEntity, _JobKey
from smartsim._core.control.jobmanager import JobManager
from smartsim._core.launcher.dragon.dragonLauncher import DragonLauncher
from smartsim._core.launcher.launcher import Launcher
from smartsim._core.launcher.local.local import LocalLauncher
from smartsim._core.launcher.lsf.lsfLauncher import LSFLauncher
from smartsim._core.launcher.pbs.pbsLauncher import PBSLauncher
from smartsim._core.launcher.slurm.slurmLauncher import SlurmLauncher
from smartsim._core.launcher.stepInfo import StepInfo
from smartsim._core.utils.helpers import get_ts_ms
from smartsim._core.utils.serialize import MANIFEST_FILENAME
from smartsim._core.utils.telemetry.collector import CollectorManager
from smartsim._core.utils.telemetry.manifest import Run, RuntimeManifest
from smartsim._core.utils.telemetry.util import map_return_code, write_event
from smartsim.error.errors import SmartSimError
from smartsim.status import TERMINAL_STATUSES

logger = logging.getLogger("TelemetryMonitor")


class ManifestEventHandler(PatternMatchingEventHandler):
    """The ManifestEventHandler monitors an experiment and updates a
    datastore as needed. This event handler is triggered by changes to
    the experiment manifest written to physical disk by a driver.

    It also contains an event loop. The loop checks experiment entities for updates
    at each timestep and executes a configurable set of metrics collectors."""

    def __init__(
        self,
        pattern: str,
        ignore_patterns: t.Optional[t.List[str]] = None,
        ignore_directories: bool = True,
        case_sensitive: bool = False,
        timeout_ms: int = 1000,
    ) -> None:
        """Initialize the manifest event handler

        :param pattern: a pattern that identifies the files whose
        events are of interest by matching their name
        :param ignore_patterns: a pattern that identifies the files whose
        events should be ignored
        :param ignore_directories: set to `True` to avoid directory events
        :param case_sensitive: set to `True` to require case sensitivity in
        resource names in order to match input patterns
        :param timeout_ms: maximum duration (in ms) of a call to the event
        loop prior to cancelling tasks
        """
        super().__init__(
            [pattern], ignore_patterns, ignore_directories, case_sensitive
        )  # type: ignore
        self._tracked_runs: t.Dict[int, Run] = {}
        self._tracked_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._completed_jobs: t.Dict[_JobKey, JobEntity] = {}
        self._launcher: t.Optional[Launcher] = None
        self.job_manager: JobManager = JobManager(threading.RLock())
        self._launcher_map: t.Dict[str, t.Type[Launcher]] = {
            "slurm": SlurmLauncher,
            "pbs": PBSLauncher,
            "lsf": LSFLauncher,
            "local": LocalLauncher,
            "dragon": DragonLauncher,
        }
        self._collector_mgr = CollectorManager(timeout_ms)

    @property
    def tracked_jobs(self) -> t.Sequence[JobEntity]:
        """The collection of `JobEntity` that are actively being monitored

        :return: the collection
        """
        return list(self._tracked_jobs.values())

    def init_launcher(self, launcher: str) -> None:
        """Initialize the controller with a specific type of launcher.
        SmartSim currently supports Slurm, PBS(Pro), LSF, Dragon
        and local launching

        :param launcher: the name of the workload manager used by the experiment
        :raises ValueError: if a string is passed that is not
        a supported launcher
        :raises TypeError: if no launcher argument is provided.
        """
        if not launcher:
            raise TypeError("Must provide a 'launcher' argument")

        if launcher_type := self._launcher_map.get(launcher.lower(), None):
            self._launcher = launcher_type()
            return

        raise ValueError("Launcher type not supported: " + launcher)

    def init_job_manager(self) -> None:
        """Initialize the job manager instance"""
        if not self._launcher:
            raise TypeError("self._launcher must be initialized")

        self.job_manager.set_launcher(self._launcher)
        self.job_manager.start()

    def set_launcher(self, launcher_type: str) -> None:
        """Set the launcher for the experiment
        :param launcher_type: the name of the workload manager used by the experiment
        """
        self.init_launcher(launcher_type)

        if self._launcher is None:
            raise SmartSimError("Launcher init failed")

        self.job_manager.set_launcher(self._launcher)
        self.job_manager.start()

    def process_manifest(self, manifest_path: str) -> None:
        """Read the manifest for the experiment. Process the
        `RuntimeManifest` by updating the set of tracked jobs
        and registered collectors

        :param manifest_path: full path to the manifest file
        """
        try:
            # it is possible to read the manifest prior to a completed
            # write due to no access locking mechanism. log the issue
            # and continue. it will retry on the next event loop iteration
            manifest = RuntimeManifest.load_manifest(manifest_path)
            if not manifest:
                logger.debug("No manifest file exists")
                return
        except json.JSONDecodeError:
            logger.error(f"Malformed manifest encountered: {manifest_path}")
            return
        except ValueError:
            logger.error("Manifest content error", exc_info=True)
            return

        if self._launcher is None:
            self.set_launcher(manifest.launcher)

        if not self._launcher:
            raise SmartSimError(f"Unable to set launcher from {manifest_path}")

        # filter out previously added items
        runs = [run for run in manifest.runs if run.timestamp not in self._tracked_runs]

        # manifest is stored at <exp_dir>/.smartsim/telemetry/manifest.json
        exp_dir = pathlib.Path(manifest_path).parent.parent.parent

        for run in runs:
            for entity in run.flatten(
                filter_fn=lambda e: e.key not in self._tracked_jobs
            ):
                entity.path = str(exp_dir)

                # track everything coming in (managed and unmanaged)
                self._tracked_jobs[entity.key] = entity

                # register collectors for new entities as needed
                if entity.telemetry_on:
                    self._collector_mgr.register_collectors(entity)

                # persist a `start` event for each new entity in the manifest
                write_event(
                    run.timestamp,
                    entity.task_id,
                    entity.step_id,
                    entity.type,
                    "start",
                    pathlib.Path(entity.status_dir),
                )

                if entity.is_managed:
                    # Tell JobManager the task is unmanaged. This collects
                    # status updates but does not try to start a new copy
                    self.job_manager.add_job(
                        entity.name,
                        entity.step_id,
                        entity,
                        False,
                    )
                    # Tell the launcher it's managed so it doesn't attempt
                    # to look for a PID that may no longer exist
                    self._launcher.step_mapping.add(
                        entity.name, entity.step_id, "", True
                    )
            self._tracked_runs[run.timestamp] = run

    def on_modified(self, event: FileSystemEvent) -> None:
        """Event handler for when a file or directory is modified.

        :param event: event representing file/directory modification.
        """
        super().on_modified(event)
        logger.debug(f"Processing manifest modified @ {event.src_path}")
        self.process_manifest(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Event handler for when a file or directory is created.

        :param event: event representing file/directory creation.
        """
        super().on_created(event)
        logger.debug(f"processing manifest created @ {event.src_path}")
        self.process_manifest(event.src_path)

    async def _to_completed(
        self,
        timestamp: int,
        entity: JobEntity,
        step_info: StepInfo,
    ) -> None:
        """Move a monitored entity from the active to completed collection to
        stop monitoring for updates during timesteps.

        :param timestamp: current timestamp for event logging
        :param entity: running SmartSim Job
        :param step_info: `StepInfo` received when requesting a Job status update
        """
        # remember completed entities to ignore them after manifest updates
        inactive_entity = self._tracked_jobs.pop(entity.key)
        if entity.key not in self._completed_jobs:
            self._completed_jobs[entity.key] = inactive_entity

        # remove all the registered collectors for the completed entity
        await self._collector_mgr.remove(entity)

        job = self.job_manager[entity.name]
        self.job_manager.move_to_completed(job)

        status_clause = f"status: {step_info.status}"
        error_clause = f", error: {step_info.error}" if step_info.error else ""

        write_path = pathlib.Path(entity.status_dir)

        # persist a `stop` event for an entity that has completed
        write_event(
            timestamp,
            entity.task_id,
            entity.step_id,
            entity.type,
            "stop",
            write_path,
            detail=f"{status_clause}{error_clause}",
            return_code=map_return_code(step_info),
        )

    async def on_timestep(self, timestamp: int) -> None:
        """Called at polling frequency to request status updates on
        monitored entities

        :param timestamp: current timestamp for event logging
        """
        if not self._launcher:
            return

        await self._collector_mgr.collect()

        # ensure unmanaged jobs move out of tracked jobs list
        u_jobs = [job for job in self._tracked_jobs.values() if not job.is_managed]
        for job in u_jobs:
            job.check_completion_status()
            if job.is_complete:
                completed_entity = self._tracked_jobs.pop(job.key)
                self._completed_jobs[job.key] = completed_entity

        # consider not using name to avoid collisions
        m_jobs = [job for job in self._tracked_jobs.values() if job.is_managed]
        if names := {entity.name: entity for entity in m_jobs}:
            step_updates: t.List[t.Tuple[str, t.Optional[StepInfo]]] = []

            try:
                task_names = list(names.keys())
                updates = self._launcher.get_step_update(task_names)
                step_updates.extend(updates)
                logger.debug(f"Retrieved updates for: {task_names}")
            except Exception:
                logger.warning(f"Telemetry step updates failed for {names.keys()}")

            try:
                for step_name, step_info in step_updates:
                    if step_info and step_info.status in TERMINAL_STATUSES:
                        completed_entity = names[step_name]
                        await self._to_completed(timestamp, completed_entity, step_info)
            except Exception as ex:
                msg = f"An error occurred getting step updates on {names}"
                logger.error(msg, exc_info=ex)

    async def shutdown(self) -> None:
        """Release all resources owned by the `ManifestEventHandler`"""
        logger.debug(f"{type(self).__name__} shutting down...")
        await self._collector_mgr.shutdown()
        logger.debug(f"{type(self).__name__} shutdown complete...")


class TelemetryMonitorArgs:
    """Strongly typed entity to house logic for validating
    configuration passed to the telemetry monitor"""

    def __init__(
        self,
        exp_dir: str,
        frequency: int,
        cooldown: int,
        log_level: int = logging.DEBUG,
    ) -> None:
        """Initialize the instance with inputs and defaults

        :param exp_dir: root path to experiment outputs
        :param frequency: desired frequency of metric & status updates (in seconds)
        :param frequency: cooldown period (in seconds) before automatic shutdown
        :param log_level: log level to apply to python logging
        """
        self.exp_dir: str = exp_dir
        self.frequency: int = frequency  # freq in seconds
        self.cooldown: int = cooldown  # cooldown in seconds
        self.log_level: int = log_level
        self._validate()

    @property
    def min_frequency(self) -> int:
        """The minimum duration (in seconds) for the monitoring loop to wait
        between executions of the monitoring loop. Shorter frequencies may
        not allow the monitoring loop to complete. Adjusting the minimum frequency
        can result in inconsistent or missing outputs due to the telemetry
        monitor cancelling processes that exceed the allotted frequency."""
        return 1

    @property
    def max_frequency(self) -> int:
        """The maximum duration (in seconds) for the monitoring loop to wait
        between executions of the monitoring loop. Longer frequencies potentially
        keep the telemetry monitor alive unnecessarily."""
        return 600

    @property
    def min_cooldown(self) -> int:
        """The minimum allowed cooldown period that can be configured. Ensures
        the cooldown does not cause the telemetry monitor to shutdown prior to
        completing a single pass through the monitoring loop"""
        return min(self.frequency + 1, self.cooldown)

    @property
    def max_cooldown(self) -> int:
        """The maximum allowed cooldown period that can be configured. Ensures the
        telemetry monitor can automatically shutdown if not needed"""
        return self.max_frequency

    @property
    def cooldown_ms(self) -> int:
        """The duration of the time period (in ms) the telemetry monitor will
        wait for new resources to monitor before shutting down"""
        return self.cooldown * 1000

    @property
    def frequency_ms(self) -> int:
        """The desired frequency (in ms) of the telemetry monitor attempts
        to retrieve status updates and metrics"""
        return self.frequency * 1000

    def _check_exp_dir(self) -> None:
        """Validate the existence of the experiment directory"""
        if not pathlib.Path(self.exp_dir).exists():
            raise ValueError(f"Experiment directory cannot be found: {self.exp_dir}")

    def _check_frequency(self) -> None:
        """Validate the frequency input is in the range
        [`min_frequency`, `max_frequency`]"""
        if self.max_frequency >= self.frequency >= self.min_frequency:
            return

        freq_tpl = "Telemetry collection frequency must be in the range [{0}, {1}]"
        raise ValueError(freq_tpl.format(self.min_frequency, self.max_frequency))

    def _check_log_level(self) -> None:
        """Validate the frequency log level input. Uses standard python log levels"""
        if self.log_level not in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ]:
            raise ValueError(f"Invalid log_level supplied: {self.log_level}")

    def _validate(self) -> None:
        """Execute all validation functions"""
        self._check_exp_dir()
        self._check_frequency()
        self._check_log_level()


class TelemetryMonitor:
    """The telemetry monitor is a standalone process managed by SmartSim to perform
    long-term retrieval of experiment status updates and resource usage
    metrics. Note that a non-blocking driver script is likely to complete before
    the SmartSim entities complete. Also, the JobManager performs status updates
    only as long as the driver is running. This telemetry monitor entrypoint is
    started automatically when a SmartSim experiment calls the `start` method
    on resources. The entrypoint runs until it has no resources to monitor."""

    def __init__(self, telemetry_monitor_args: TelemetryMonitorArgs):
        """Initialize the telemetry monitor instance

        :param telemetry_monitor_args: configuration for the telemetry monitor
        """
        self._observer: BaseObserver = Observer()
        """an observer object that triggers the action handler"""
        self._args = telemetry_monitor_args
        """user-supplied arguments configuring telemetry monitor behavior"""
        self._experiment_dir = pathlib.Path(self._args.exp_dir)
        """path to the root directory where experiment outputs are written"""
        self._telemetry_path = self._experiment_dir / CONFIG.telemetry_subdir
        """path to the root directory where telemetry outputs are written"""
        self._manifest_path = self._telemetry_path / MANIFEST_FILENAME
        """path to the runtime manifest file"""
        self._action_handler: t.Optional[ManifestEventHandler] = None
        """an event listener holding action handlers for manifest on-change events"""

    def _can_shutdown(self) -> bool:
        """Determines if the telemetry monitor can perform shutdown. An
        automatic shutdown will occur if there are no active jobs being monitored.
        Managed jobs and databases are considered separately due to the way they
        are stored in the job manager

        :return: return True if capable of automatically shutting down
        """
        managed_jobs = (
            list(self._action_handler.job_manager.jobs.values())
            if self._action_handler
            else []
        )
        unmanaged_jobs = (
            list(self._action_handler.tracked_jobs) if self._action_handler else []
        )
        # get an individual count of databases for logging
        n_dbs: int = len(
            [
                job
                for job in managed_jobs + unmanaged_jobs
                if isinstance(job, JobEntity) and job.is_db
            ]
        )

        # if we have no jobs currently being monitored we can shutdown
        n_jobs = len(managed_jobs) + len(unmanaged_jobs) - n_dbs
        shutdown_ok = n_jobs + n_dbs == 0

        logger.debug(f"{n_jobs} active job(s), {n_dbs} active db(s)")
        return shutdown_ok

    async def monitor(self) -> None:
        """The main monitoring loop. Executes a busy wait and triggers
        telemetry collectors using frequency from constructor arguments.
        Continue monitoring until it satisfies automatic shutdown criteria."""
        elapsed: int = 0
        last_ts: int = get_ts_ms()
        shutdown_in_progress = False

        if self._action_handler is None:
            raise ValueError("The action handler must be initialized to monitor")

        # Event loop runs until the observer shuts down or
        # an automatic shutdown is started.
        while self._observer.is_alive() and not shutdown_in_progress:
            duration_ms = 0
            start_ts = get_ts_ms()
            await self._action_handler.on_timestep(start_ts)

            elapsed += start_ts - last_ts
            last_ts = start_ts

            # check if there are no jobs being monitored
            if self._can_shutdown():
                # cooldown period begins accumulating when no entities are monitored
                if elapsed >= self._args.cooldown_ms:
                    shutdown_in_progress = True
                    logger.info("Cooldown complete. Beginning shutdown")
                    await self._action_handler.shutdown()
                    logger.debug("Beginning file monitor shutdown")
                    self._observer.stop()  # type: ignore
                    logger.debug("Event loop shutdown complete")
                    break
            else:
                # reset cooldown any time jobs are running
                elapsed = 0

            # track time elapsed to execute metric collection
            duration_ms = get_ts_ms() - start_ts
            wait_ms = max(self._args.frequency_ms - duration_ms, 0)

            # delay next loop if collection time didn't exceed loop frequency
            wait_sec = wait_ms / 1000  # convert to seconds for sleep
            if elapsed > 0:
                completion_pct = elapsed / self._args.cooldown_ms * 100
                logger.info(f"Cooldown {completion_pct:.2f}% complete")
            logger.debug(f"Collection in {wait_sec:.2f}s")
            await asyncio.sleep(wait_sec)

        logger.info("Exiting telemetry monitor event loop")

    async def run(self) -> int:
        """Setup the monitoring entities and start the timer-based loop that
        will poll for telemetry data

        :return: return code for the process
        """
        logger.info("Executing telemetry monitor")
        logger.info(f"Polling frequency: {self._args.frequency}s")
        logger.info(f"Experiment directory: {self._experiment_dir}")
        logger.info(f"Telemetry output: {self._telemetry_path}")

        # Convert second-based inputs to milliseconds
        frequency_ms = int(self._args.frequency * 1000)

        # Create event handlers to trigger when target files are changed
        log_handler = LoggingEventHandler(logger)
        self._action_handler = ManifestEventHandler(
            str(MANIFEST_FILENAME),
            timeout_ms=frequency_ms,
            ignore_patterns=["*.out", "*.err"],
        )

        try:
            # The manifest may not exist when the telemetry monitor starts
            if self._manifest_path.exists():
                self._action_handler.process_manifest(str(self._manifest_path))

            # Add a handler to log file-system events
            self._observer.schedule(log_handler, self._telemetry_path)  # type:ignore
            # Add a handler to perform actions on file-system events
            self._observer.schedule(
                self._action_handler, self._telemetry_path
            )  # type:ignore
            self._observer.start()  # type: ignore

            # kick off the 'infinite' monitoring loop
            await self.monitor()
            return os.EX_OK
        except Exception as ex:
            logger.error(ex)
        finally:
            await self._action_handler.shutdown()
            self.cleanup()
            logger.info("Telemetry monitor shutdown complete")

        return os.EX_SOFTWARE

    def cleanup(self) -> None:
        """Perform cleanup for all allocated resources"""
        if self._observer is not None and self._observer.is_alive():
            logger.debug("Cleaning up manifest observer")
            self._observer.stop()  # type: ignore
            self._observer.join()
