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

# pylint: disable=too-many-lines

from __future__ import annotations

import collections
import datetime
import itertools
import os
import os.path as osp
import pathlib
import textwrap
import typing as t
from os import environ, getcwd

from tabulate import tabulate

from smartsim._core import dispatch
from smartsim._core.config import CONFIG
from smartsim._core.control.launch_history import LaunchHistory as _LaunchHistory
from smartsim.error import errors
from smartsim.status import InvalidJobStatus, JobStatus

from ._core import Controller, Generator, Manifest, previewrenderer
from .database import FeatureStore
from .entity import (
    Application,
    Ensemble,
    EntitySequence,
    SmartSimEntity,
    TelemetryConfiguration,
)
from .error import SmartSimError
from .log import ctx_exp_path, get_logger, method_contextualizer

if t.TYPE_CHECKING:
    from smartsim._core.utils.launcher import ExecutableProtocol
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
    """Experiment is a factory class that creates stages of a workflow
    and manages their execution.

    The instances created by an Experiment represent executable code
    that is either user-specified, like the ``Application`` instance created
    by ``Experiment.create_application``, or pre-configured, like the ``FeatureStore``
    instance created by ``Experiment.create_feature_store``.

    Experiment methods that accept a variable list of arguments, such as
    ``Experiment.start`` or ``Experiment.stop``, accept any number of the
    instances created by the Experiment.

    In general, the Experiment class is designed to be initialized once
    and utilized throughout runtime.
    """

    def __init__(self, name: str, exp_path: str | None = None):
        """Initialize an Experiment instance.

        With the default settings, the Experiment will use the
        local launcher, which will start all Experiment created
        instances on the localhost.

        Example of initializing an Experiment

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="local")

        SmartSim supports multiple launchers which also can be specified
        based on the type of system you are running on.

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="slurm")

        If you want your Experiment driver script to be run across
        multiple system with different schedulers (workload managers)
        you can also use the `auto` argument to have the Experiment detect
        which launcher to use based on system installed binaries and libraries.

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="auto")


        The Experiment path will default to the current working directory
        and if the ``Experiment.generate`` method is called, a directory
        with the Experiment name will be created to house the output
        from the Experiment.

        :param name: name for the ``Experiment``
        :param exp_path: path to location of ``Experiment`` directory
        """
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

    def start(self, *jobs: Job) -> tuple[LaunchedJobID, ...]:
        """Execute a collection of `Job` instances.

        :param jobs: A collection of other job instances to start
        :returns: A sequence of ids with order corresponding to the sequence of
            jobs that can be used to query or alter the status of that
            particular execution of the job.
        """
        # Create the run id
        run_id = datetime.datetime.now().replace(microsecond=0).isoformat()
        # Generate the root path
        root = pathlib.Path(self.exp_path, run_id)
        return self._dispatch(Generator(root), dispatch.DEFAULT_DISPATCHER, *jobs)

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
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # FIXME: Remove this cast after `SmartSimEntity` conforms to
            #        protocol. For now, live with the "dangerous" type cast
            # ---------------------------------------------------------------------
            exe = t.cast("ExecutableProtocol", job.entity)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            job_execution_path, out, err = self._generate(generator, job, idx)
            id_ = launch_config.start(exe, job_execution_path, env, out, err)
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
        :returns: A tuple of statuses with order respective of the order of the
            calling arguments.
        """
        to_query = self._launch_history.group_by_launcher(
            set(ids), unknown_ok=True
        ).items()
        stats_iter = (launcher.get_status(*ids).items() for launcher, ids in to_query)
        stats_map = dict(itertools.chain.from_iterable(stats_iter))
        stats = (stats_map.get(i, InvalidJobStatus.NEVER_STARTED) for i in ids)
        return tuple(stats)

    @_contextualize
    def _generate(
        self, generator: Generator, job: Job, job_index: int
    ) -> t.Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """Generate the directory structure and files for a ``Job``

        If files or directories are attached to an ``Application`` object
        associated with the Job using ``Application.attach_generator_files()``,
        those files or directories will be symlinked, copied, or configured and
        written into the created job directory.

        :param generator: The generator is responsible for creating the job
            run and log directory.
        :param job: The Job instance for which the output is generated.
        :param job_index: The index of the Job instance (used for naming).
        :returns: The path to the generated output for the Job instance.
        :raises: A SmartSimError if an error occurs during the generation process.
        """
        try:
            job_path, out, err = generator.generate_job(job, job_index)
            return (job_path, out, err)
        except SmartSimError as e:
            logger.error(e)
            raise

    def preview(
        self,
        *args: t.Any,
        verbosity_level: previewrenderer.Verbosity = previewrenderer.Verbosity.INFO,
        output_format: previewrenderer.Format = previewrenderer.Format.PLAINTEXT,
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

        previewrenderer.render(
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
