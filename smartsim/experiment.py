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

import os
import os.path as osp
import typing as t
from os import environ, getcwd

from tabulate import tabulate

from smartsim._core.config import CONFIG
from smartsim.error.errors import SSUnsupportedError
from smartsim.status import SmartSimStatus

from ._core import Controller, Generator, Manifest
from .database import Orchestrator
from .entity import (
    Ensemble,
    EntitySequence,
    Model,
    SmartSimEntity,
    TelemetryConfiguration,
)
from .error import SmartSimError
from .log import ctx_exp_path, get_logger, method_contextualizer
from .settings import Container, base, settings
from .wlm import detect_launcher

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
    that is either user-specified, like the ``Model`` instance created
    by ``Experiment.create_model``, or pre-configured, like the ``Orchestrator``
    instance created by ``Experiment.create_database``.

    Experiment methods that accept a variable list of arguments, such as
    ``Experiment.start`` or ``Experiment.stop``, accept any number of the
    instances created by the Experiment.

    In general, the Experiment class is designed to be initialized once
    and utilized throughout runtime.
    """

    def __init__(
        self,
        name: str,
        exp_path: t.Optional[str] = None,
        launcher: str = "local",
    ):
        """Initialize an Experiment instance.

        With the default settings, the Experiment will use the
        local launcher, which will start all Experiment created
        instances on the localhost.

        Example of initializing an Experiment with the local launcher

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
        :param launcher: type of launcher being used, options are "slurm", "pbs",
                         "lsf", or "local". If set to "auto",
                         an attempt will be made to find an available launcher
                         on the system.
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

        if launcher == "auto":
            launcher = detect_launcher()
        if launcher == "cobalt":
            raise SSUnsupportedError("Cobalt launcher is no longer supported.")

        self._control = Controller(launcher=launcher)
        self._launcher = launcher.lower()
        self.db_identifiers: t.Set[str] = set()
        self._telemetry_cfg = ExperimentTelemetryConfiguration()

    @_contextualize
    def start(
        self,
        *args: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]],
        block: bool = True,
        summary: bool = False,
        kill_on_interrupt: bool = True,
    ) -> None:
        """Start passed instances using Experiment launcher

        Any instance ``Model``, ``Ensemble`` or ``Orchestrator``
        instance created by the Experiment can be passed as
        an argument to the start method.

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="slurm")
            settings = exp.create_run_settings(exe="./path/to/binary")
            model = exp.create_model("my_model", settings)
            exp.start(model)

        Multiple entity instances can also be passed to the start method
        at once no matter which type of instance they are. These will
        all be launched together.

        .. highlight:: python
        .. code-block:: python

            exp.start(model_1, model_2, db, ensemble, block=True)
            # alternatively
            stage_1 = [model_1, model_2, db, ensemble]
            exp.start(*stage_1, block=True)


        If `block==True` the Experiment will poll the launched instances
        at runtime until all non-database jobs have completed. Database
        jobs *must* be killed by the user by passing them to
        ``Experiment.stop``. This allows for multiple stages of a workflow
        to produce to and consume from the same Orchestrator database.

        If `kill_on_interrupt=True`, then all jobs launched by this
        experiment are guaranteed to be killed when ^C (SIGINT) signal is
        received. If `kill_on_interrupt=False`, then it is not guaranteed
        that all jobs launched by this experiment will be killed, and the
        zombie processes will need to be manually killed.

        :param block: block execution until all non-database
                       jobs are finished
        :param summary: print a launch summary prior to launch
        :param kill_on_interrupt: flag for killing jobs when ^C (SIGINT)
                                  signal is received.
        """
        start_manifest = Manifest(*args)
        self._create_entity_dir(start_manifest)
        try:
            if summary:
                self._launch_summary(start_manifest)
            self._control.start(
                exp_name=self.name,
                exp_path=self.exp_path,
                manifest=start_manifest,
                block=block,
                kill_on_interrupt=kill_on_interrupt,
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def stop(
        self, *args: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> None:
        """Stop specific instances launched by this ``Experiment``

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the stop method.

        Whichever launcher was specified at Experiment initialization
        will be used to stop the instance. For example, which using
        the slurm launcher, this equates to running `scancel` on the
        instance.

        Example

        .. highlight:: python
        .. code-block:: python

            exp.stop(model)
            # multiple
            exp.stop(model_1, model_2, db, ensemble)

        :param args: One or more SmartSimEntity or EntitySequence objects.
        :raises TypeError: if wrong type
        :raises SmartSimError: if stop request fails
        """
        stop_manifest = Manifest(*args)
        try:
            for entity in stop_manifest.models:
                self._control.stop_entity(entity)
            for entity_list in stop_manifest.ensembles:
                self._control.stop_entity_list(entity_list)
            dbs = stop_manifest.dbs
            for db in dbs:
                self._control.stop_db(db)
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def generate(
        self,
        *args: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]],
        tag: t.Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """Generate the file structure for an ``Experiment``

        ``Experiment.generate`` creates directories for each entity
        passed to organize Experiments that launch many entities.

        If files or directories are attached to ``Model`` objects
        using ``Model.attach_generator_files()``, those files or
        directories will be symlinked, copied, or configured and
        written into the created directory for that instance.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the generate method.

        :param tag: tag used in `to_configure` generator files
        :param overwrite: overwrite existing folders and contents
        :param verbose: log parameter settings to std out
        """
        try:
            generator = Generator(self.exp_path, overwrite=overwrite, verbose=verbose)
            if tag:
                generator.set_tag(tag)
            generator.generate_experiment(*args)
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def poll(
        self, interval: int = 10, verbose: bool = True, kill_on_interrupt: bool = True
    ) -> None:
        """Monitor jobs through logging to stdout.

        This method should only be used if jobs were launched
        with ``Experiment.start(block=False)``

        The internal specified will control how often the
        logging is performed, not how often the polling occurs.
        By default, internal polling is set to every second for
        local launcher jobs and every 10 seconds for all other
        launchers.

        If internal polling needs to be slower or faster based on
        system or site standards, set the ``SMARTSIM_JM_INTERNAL``
        environment variable to control the internal polling interval
        for SmartSim.

        For more verbose logging output, the ``SMARTSIM_LOG_LEVEL``
        environment variable can be set to `debug`

        If `kill_on_interrupt=True`, then all jobs launched by this
        experiment are guaranteed to be killed when ^C (SIGINT) signal is
        received. If `kill_on_interrupt=False`, then it is not guaranteed
        that all jobs launched by this experiment will be killed, and the
        zombie processes will need to be manually killed.

        :param interval: frequency (in seconds) of logging to stdout
        :param verbose: set verbosity
        :param kill_on_interrupt: flag for killing jobs when SIGINT is received
        :raises SmartSimError: if poll request fails
        """
        try:
            self._control.poll(interval, verbose, kill_on_interrupt=kill_on_interrupt)
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def finished(self, entity: SmartSimEntity) -> bool:
        """Query if a job has completed.

        An instance of ``Model`` or ``Ensemble`` can be passed
        as an argument.

        Passing ``Orchestrator`` will return an error as a
        database deployment is never finished until stopped
        by the user.

        :param entity: object launched by this ``Experiment``
        :returns: True if the job has finished, False otherwise
        :raises SmartSimError: if entity has not been launched
                               by this ``Experiment``
        """
        try:
            return self._control.finished(entity)
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def get_status(
        self, *args: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]]
    ) -> t.List[SmartSimStatus]:
        """Query the status of launched entity instances

        Return a smartsim.status string representing
        the status of the launched instance.

        .. highlight:: python
        .. code-block:: python

            exp.get_status(model)

        As with an Experiment method, multiple instance of
        varying types can be passed to and all statuses will
        be returned at once.

        .. highlight:: python
        .. code-block:: python

            statuses = exp.get_status(model, ensemble, orchestrator)
            complete = [s == smartsim.status.STATUS_COMPLETED for s in statuses]
            assert all(complete)

        :returns: status of the instances passed as arguments
        :raises SmartSimError: if status retrieval fails
        """
        try:
            manifest = Manifest(*args)
            statuses: t.List[SmartSimStatus] = []
            for entity in manifest.models:
                statuses.append(self._control.get_entity_status(entity))
            for entity_list in manifest.all_entity_lists:
                statuses.extend(self._control.get_entity_list_status(entity_list))
            return statuses
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def reconnect_orchestrator(self, checkpoint: str) -> Orchestrator:
        """Reconnect to a running ``Orchestrator``

        This method can be used to connect to a ``Orchestrator`` deployment
        that was launched by a previous ``Experiment``. This can be
        helpful in the case where separate runs of an ``Experiment``
        wish to use the same ``Orchestrator`` instance currently
        running on a system.

        :param checkpoint: the `smartsim_db.dat` file created
                           when an ``Orchestrator`` is launched
        """
        try:
            orc = self._control.reload_saved_db(checkpoint)
            return orc
        except SmartSimError as e:
            logger.error(e)
            raise

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
        values = []
        headers = [
            "Name",
            "Entity-Type",
            "JobID",
            "RunID",
            "Time",
            "Status",
            "Returncode",
        ]
        for job in self._control.get_jobs().values():
            for run in range(job.history.runs + 1):
                values.append(
                    [
                        job.entity.name,
                        job.entity.type,
                        job.history.jids[run],
                        run,
                        f"{job.history.job_times[run]:.4f}",
                        job.history.statuses[run],
                        job.history.returns[run],
                    ]
                )
        return tabulate(
            values,
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

    def _launch_summary(self, manifest: Manifest) -> None:
        """Experiment pre-launch summary of entities that will be launched

        :param manifest: Manifest of deployables.
        """

        summary = "\n\n=== Launch Summary ===\n"
        summary += f"Experiment: {self.name}\n"
        summary += f"Experiment Path: {self.exp_path}\n"
        summary += f"Launcher: {self._launcher}\n"
        if manifest.models:
            summary += f"Models: {len(manifest.models)}\n"

        if self._control.orchestrator_active:
            summary += "Database Status: active\n"
        elif manifest.dbs:
            summary += "Database Status: launching\n"
        else:
            summary += "Database Status: inactive\n"

        summary += f"\n{str(manifest)}"

        logger.info(summary)

    def _create_entity_dir(self, start_manifest: Manifest) -> None:
        def create_entity_dir(entity: t.Union[Orchestrator, Model, Ensemble]) -> None:
            if not os.path.isdir(entity.path):
                os.makedirs(entity.path)

        for model in start_manifest.models:
            create_entity_dir(model)

        for orch in start_manifest.dbs:
            create_entity_dir(orch)

        for ensemble in start_manifest.ensembles:
            create_entity_dir(ensemble)

            for member in ensemble.models:
                create_entity_dir(member)

    def __str__(self) -> str:
        return self.name

    def _append_to_db_identifier_list(self, db_identifier: str) -> None:
        """Check if db_identifier already exists when calling create_database"""
        if db_identifier in self.db_identifiers:
            logger.warning(
                f"A database with the identifier {db_identifier} has already been made "
                "An error will be raised if multiple databases are started "
                "with the same identifier"
            )
        # Otherwise, add
        self.db_identifiers.add(db_identifier)
