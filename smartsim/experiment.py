# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

from __future__ import annotations

import datetime
import itertools
import os.path as osp
import pathlib
import typing as t
from os import environ, getcwd

from tabulate import tabulate

from smartsim._core import dispatch
from smartsim._core.config import CONFIG
from smartsim._core.control import interval as _interval
from smartsim._core.control import preview_renderer
from smartsim._core.control.launch_history import LaunchHistory as _LaunchHistory
from smartsim._core.utils import helpers as _helpers
from smartsim.error import errors
from smartsim.launchable.job import Job
from smartsim.status import TERMINAL_STATUSES, InvalidJobStatus, JobStatus

from ._core import Generator, Manifest
from ._core.generation.generator import Job_Path
from .entity import TelemetryConfiguration
from .error import SmartSimError
from .log import ctx_exp_path, get_logger, method_contextualizer

if t.TYPE_CHECKING:
    from smartsim.launchable.job import Job
    from smartsim.types import LaunchedJobID

logger = get_logger(__name__)


def _exp_path_map(exp: "Experiment") -> str:
    """Mapping function for use by method contextualizer to place the path of
    the currently-executing experiment into context for log enrichment"""
    return exp.exp_path


_contextualize = method_contextualizer(ctx_exp_path, _exp_path_map)


class ExperimentTelemetryConfiguration(TelemetryConfiguration):
    """Customized telemetry configuration for an `Experiment`. Ensures
    backwards compatible behavior with drivers using environment variables
    to enable experiment telemetry"""

    def __init__(self) -> None:
        super().__init__(enabled=CONFIG.telemetry_enabled)

    def _on_enable(self) -> None:
        """Modify the environment variable to enable telemetry."""
        environ["SMARTSIM_FLAG_TELEMETRY"] = "1"

    def _on_disable(self) -> None:
        """Modify the environment variable to disable telemetry."""
        environ["SMARTSIM_FLAG_TELEMETRY"] = "0"


# pylint: disable=no-self-use
class Experiment:
    """The Experiment class is used to schedule, launch, track, and manage
    jobs and job groups.  Also, it is the SmartSim class that manages
    internal data structures, processes, and infrastructure for interactive
    capabilities such as the SmartSim dashboard and historical lookback on
    launched jobs and job groups.  The Experiment class is designed to be
    initialized once and utilized throughout the entirety of a workflow.
    """

    def __init__(self, name: str, exp_path: str | None = None):
        """Initialize an Experiment instance.


        Example of initializing an Experiment


        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp")

        The name of a SmartSim ``Experiment`` will determine the
        name of the ``Experiment`` directory that is created inside of the
        current working directory.

        If a different ``Experiment`` path is desired, the ``exp_path``
        parameter can be set as shown in the example below.

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", exp_path="/full/path/to/exp")


        Note that the provided path must exist prior to ``Experiment``
        construction and that an experiment name subdirectory will not be
        created inside of the provide path.

        :param name: name for the ``Experiment``
        :param exp_path: path to location of ``Experiment`` directory
        """

        if name:
            if not isinstance(name, str):
                raise TypeError("name argument was not of type str")
        else:
            raise TypeError("Experiment name must be non-empty string")

        self.name = name

        if exp_path:
            if not isinstance(exp_path, str):
                raise TypeError("exp_path argument was not of type str")
            if not osp.isdir(osp.abspath(exp_path)):
                raise NotADirectoryError("Experiment path provided does not exist")
            exp_path = osp.abspath(exp_path)
        else:
            exp_path = osp.join(getcwd(), name)

        self.exp_path = exp_path
        """The path under which the experiment operate"""

        self._launch_history = _LaunchHistory()
        """A cache of launchers used and which ids they have issued"""
        self._fs_identifiers: t.Set[str] = set()
        """Set of feature store identifiers currently in use by this
        experiment
        """
        self._telemetry_cfg = ExperimentTelemetryConfiguration()
        """Switch to specify if telemetry data should be produced for this
        experiment
        """

    def _set_dragon_server_path(self) -> None:
        """Set path for dragon server through environment varialbes"""
        if not "SMARTSIM_DRAGON_SERVER_PATH" in environ:
            environ["_SMARTSIM_DRAGON_SERVER_PATH_EXP"] = osp.join(
                self.exp_path, CONFIG.dragon_default_subdir
            )

    def start(self, *jobs: Job | t.Sequence[Job]) -> tuple[LaunchedJobID, ...]:
        """Execute a collection of `Job` instances.

        :param jobs: A collection of other job instances to start
        :raises TypeError: If jobs provided are not the correct type
        :raises ValueError: No Jobs were provided.
        :returns: A sequence of ids with order corresponding to the sequence of
            jobs that can be used to query or alter the status of that
            particular execution of the job.
        """

        if not jobs:
            raise ValueError("No jobs provided to start")

        # Create the run id
        jobs_ = list(_helpers.unpack(jobs))

        run_id = datetime.datetime.now().replace(microsecond=0).isoformat()
        root = pathlib.Path(self.exp_path, run_id.replace(":", "."))
        return self._dispatch(Generator(root), dispatch.DEFAULT_DISPATCHER, *jobs_)

    def _dispatch(
        self,
        generator: Generator,
        dispatcher: dispatch.Dispatcher,
        job: Job,
        *jobs: Job,
    ) -> tuple[LaunchedJobID, ...]:
        """Dispatch a series of jobs with a particular dispatcher

        :param generator: The generator is responsible for creating the
            job run and log directory.
        :param dispatcher: The dispatcher that should be used to determine how
            to start a job based on its launch settings.
        :param job: The first job instance to dispatch
        :param jobs: A collection of other job instances to dispatch
        :returns: A sequence of ids with order corresponding to the sequence of
            jobs that can be used to query or alter the status of that
            particular dispatch of the job.
        """

        def execute_dispatch(generator: Generator, job: Job, idx: int) -> LaunchedJobID:
            args = job.launch_settings.launch_args
            env = job.launch_settings.env_vars
            exe = job.entity.as_executable_sequence()
            dispatch = dispatcher.get_dispatch(args)
            try:
                # Check to see if one of the existing launchers can be
                # configured to handle the launch arguments ...
                launch_config = dispatch.configure_first_compatible_launcher(
                    from_available_launchers=self._launch_history.iter_past_launchers(),
                    with_arguments=args,
                )
            except errors.LauncherNotFoundError:
                # ... otherwise create a new launcher that _can_ handle the
                # launch arguments and configure _that_ one
                launch_config = dispatch.create_new_launcher_configuration(
                    for_experiment=self, with_arguments=args
                )
            # Generate the job directory and return the generated job path
            job_paths = self._generate(generator, job, idx)
            id_ = launch_config.start(
                exe, job_paths.run_path, env, job_paths.out_path, job_paths.err_path
            )
            # Save the underlying launcher instance and launched job id. That
            # way we do not need to spin up a launcher instance for each
            # individual job, and the experiment can monitor job statuses.
            # pylint: disable-next=protected-access
            self._launch_history.save_launch(launch_config._adapted_launcher, id_)
            return id_

        return execute_dispatch(generator, job, 0), *(
            execute_dispatch(generator, job, idx) for idx, job in enumerate(jobs, 1)
        )

    def get_status(
        self, *ids: LaunchedJobID
    ) -> tuple[JobStatus | InvalidJobStatus, ...]:
        """Get the status of jobs launched through the `Experiment` from their
        launched job id returned when calling `Experiment.start`.

        The `Experiment` will map the launched ID back to the launcher that
        started the job and request a status update. The order of the returned
        statuses exactly matches the order of the launched job ids.

        If the `Experiment` cannot find any launcher that started the job
        associated with the launched job id, then a
        `InvalidJobStatus.NEVER_STARTED` status is returned for that id.

        If the experiment maps the launched job id to multiple launchers, then
        a `ValueError` is raised. This should only happen in the case when
        launched job ids issued by user defined launcher are not sufficiently
        unique.

        :param ids: A sequence of launched job ids issued by the experiment.
        :raises TypeError: If ids provided are not the correct type
        :raises ValueError: No IDs were provided.
        :returns: A tuple of statuses with order respective of the order of the
            calling arguments.
        """
        if not ids:
            raise ValueError("No job ids provided to get status")
        if not all(isinstance(id, str) for id in ids):
            raise TypeError("ids argument was not of type LaunchedJobID")

        to_query = self._launch_history.group_by_launcher(
            set(ids), unknown_ok=True
        ).items()
        stats_iter = (launcher.get_status(*ids).items() for launcher, ids in to_query)
        stats_map = dict(itertools.chain.from_iterable(stats_iter))
        stats = (stats_map.get(i, InvalidJobStatus.NEVER_STARTED) for i in ids)
        return tuple(stats)

    def wait(
        self, *ids: LaunchedJobID, timeout: float | None = None, verbose: bool = True
    ) -> None:
        """Block execution until all of the provided launched jobs, represented
        by an ID, have entered a terminal status.

        :param ids: The ids of the launched jobs to wait for.
        :param timeout: The max time to wait for all of the launched jobs to end.
        :param verbose: Whether found statuses should be displayed in the console.
        :raises TypeError: If IDs provided are not the correct type
        :raises ValueError: No IDs were provided.
        """
        if ids:
            if not all(isinstance(id, str) for id in ids):
                raise TypeError("ids argument was not of type LaunchedJobID")
        else:
            raise ValueError("No job ids to wait on provided")
        self._poll_for_statuses(
            ids, TERMINAL_STATUSES, timeout=timeout, verbose=verbose
        )

    def _poll_for_statuses(
        self,
        ids: t.Sequence[LaunchedJobID],
        statuses: t.Collection[JobStatus],
        timeout: float | None = None,
        interval: float = 5.0,
        verbose: bool = True,
    ) -> dict[LaunchedJobID, JobStatus | InvalidJobStatus]:
        """Poll the experiment's launchers for the statuses of the launched
        jobs with the provided ids, until the status of the changes to one of
        the provided statuses.

        :param ids: The ids of the launched jobs to wait for.
        :param statuses: A collection of statuses to poll for.
        :param timeout: The minimum amount of time to spend polling all jobs to
            reach one of the supplied statuses. If not supplied or `None`, the
            experiment will poll indefinitely.
        :param interval: The minimum time between polling launchers.
        :param verbose: Whether or not to log polled states to the console.
        :raises ValueError: The interval between polling launchers is infinite
        :raises TimeoutError: The polling interval was exceeded.
        :returns: A mapping of ids to the status they entered that ended
            polling.
        """
        terminal = frozenset(itertools.chain(statuses, InvalidJobStatus))
        log = logger.info if verbose else lambda *_, **__: None
        method_timeout = _interval.SynchronousTimeInterval(timeout)
        iter_timeout = _interval.SynchronousTimeInterval(interval)
        final: dict[LaunchedJobID, JobStatus | InvalidJobStatus] = {}

        def is_finished(
            id_: LaunchedJobID, status: JobStatus | InvalidJobStatus
        ) -> bool:
            job_title = f"Job({id_}): "
            if done := status in terminal:
                log(f"{job_title}Finished with status '{status.value}'")
            else:
                log(f"{job_title}Running with status '{status.value}'")
            return done

        if iter_timeout.infinite:
            raise ValueError("Polling interval cannot be infinite")
        while ids and not method_timeout.expired:
            iter_timeout = iter_timeout.new_interval()
            stats = zip(ids, self.get_status(*ids))
            is_done = _helpers.group_by(_helpers.pack_params(is_finished), stats)
            final |= dict(is_done.get(True, ()))
            ids = tuple(id_ for id_, _ in is_done.get(False, ()))
            if ids:
                (
                    iter_timeout
                    if iter_timeout.remaining < method_timeout.remaining
                    else method_timeout
                ).block()
        if ids:
            raise TimeoutError(
                f"Job ID(s) {', '.join(map(str, ids))} failed to reach "
                "terminal status before timeout"
            )
        return final

    @_contextualize
    def _generate(self, generator: Generator, job: Job, job_index: int) -> Job_Path:
        """Generate the directory structure and files for a ``Job``

        If files or directories are attached to an ``Application`` object
        associated with the Job using ``Application.attach_generator_files()``,
        those files or directories will be symlinked, copied, or configured and
        written into the created job directory.

        :param generator: The generator is responsible for creating the job
            run and log directory.
        :param job: The Job instance for which the output is generated.
        :param job_index: The index of the Job instance (used for naming).
        :returns: The paths to the generated output for the Job instance.
        :raises: A SmartSimError if an error occurs during the generation process.
        """
        try:
            job_paths = generator.generate_job(job, job_index)
            return job_paths
        except SmartSimError as e:
            logger.error(e)
            raise

    def preview(
        self,
        *args: t.Any,
        verbosity_level: preview_renderer.Verbosity = preview_renderer.Verbosity.INFO,
        output_format: preview_renderer.Format = preview_renderer.Format.PLAINTEXT,
        output_filename: t.Optional[str] = None,
    ) -> None:
        """Preview entity information prior to launch. This method
        aggregates multiple pieces of information to give users insight
        into what and how entities will be launched.  Any instance of
        ``Model``, ``Ensemble``, or ``Feature Store`` created by the
        Experiment can be passed as an argument to the preview method.

        Verbosity levels:
         - info: Display user-defined fields and entities.
         - debug: Display user-defined field and entities and auto-generated
            fields.
         - developer: Display user-defined field and entities, auto-generated
            fields, and run commands.

        :param verbosity_level: verbosity level specified by user, defaults to info.
        :param output_format: Set output format. The possible accepted
            output formats are ``plain_text``.
            Defaults to ``plain_text``.
        :param output_filename: Specify name of file and extension to write
            preview data to. If no output filename is set, the preview will be
            output to stdout. Defaults to None.
        """

        preview_manifest = Manifest(*args)

        preview_renderer.render(
            self,
            preview_manifest,
            verbosity_level,
            output_format,
            output_filename,
        )

    @_contextualize
    def summary(self, style: str = "github") -> str:
        """Return a summary of the ``Experiment``

        The summary will show each instance that has been
        launched and completed in this ``Experiment``

        :param style: the style in which the summary table is formatted,
                       for a full list of styles see the table-format section of:
                       https://github.com/astanin/python-tabulate
        :return: tabulate string of ``Experiment`` history
        """
        headers = [
            "Name",
            "Entity-Type",
            "JobID",
            "RunID",
            "Time",
            "Status",
            "Returncode",
        ]
        return tabulate(
            [],
            headers,
            showindex=True,
            tablefmt=style,
            missingval="None",
            disable_numparse=True,
        )

    def stop(self, *ids: LaunchedJobID) -> tuple[JobStatus | InvalidJobStatus, ...]:
        """Cancel the execution of a previously launched job.

        :param ids: The ids of the launched jobs to stop.
        :raises TypeError: If ids provided are not the correct type
        :raises ValueError: No job ids were provided.
        :returns: A tuple of job statuses upon cancellation with order
            respective of the order of the calling arguments.
        """
        if ids:
            if not all(isinstance(id, str) for id in ids):
                raise TypeError("ids argument was not of type LaunchedJobID")
        else:
            raise ValueError("No job ids provided")
        by_launcher = self._launch_history.group_by_launcher(set(ids), unknown_ok=True)
        id_to_stop_stat = (
            launcher.stop_jobs(*launched).items()
            for launcher, launched in by_launcher.items()
        )
        stats_map = dict(itertools.chain.from_iterable(id_to_stop_stat))
        stats = (stats_map.get(id_, InvalidJobStatus.NEVER_STARTED) for id_ in ids)
        return tuple(stats)

    @property
    def telemetry(self) -> TelemetryConfiguration:
        """Return the telemetry configuration for this entity.

        :returns: configuration of telemetry for this entity
        """
        return self._telemetry_cfg

    def __str__(self) -> str:
        return self.name

    def _append_to_fs_identifier_list(self, fs_identifier: str) -> None:
        """Check if fs_identifier already exists when calling create_feature_store"""
        if fs_identifier in self._fs_identifiers:
            logger.warning(
                f"A feature store with the identifier {fs_identifier} has already been made "
                "An error will be raised if multiple Feature Stores are started "
                "with the same identifier"
            )
        # Otherwise, add
        self._fs_identifiers.add(fs_identifier)
