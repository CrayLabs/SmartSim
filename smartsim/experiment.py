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
    def create_ensemble(
        self,
        name: str,
        params: t.Optional[t.Dict[str, t.Any]] = None,
        batch_settings: t.Optional[base.BatchSettings] = None,
        run_settings: t.Optional[base.RunSettings] = None,
        replicas: t.Optional[int] = None,
        perm_strategy: str = "all_perm",
        path: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> Ensemble:
        """Create an ``Ensemble`` of ``Model`` instances

        Ensembles can be launched sequentially or as a batch
        if using a non-local launcher. e.g. slurm

        Ensembles require one of the following combinations
        of arguments:

            - ``run_settings`` and ``params``
            - ``run_settings`` and ``replicas``
            - ``batch_settings``
            - ``batch_settings``, ``run_settings``, and ``params``
            - ``batch_settings``, ``run_settings``, and ``replicas``

        If given solely batch settings, an empty ensemble
        will be created that Models can be added to manually
        through ``Ensemble.add_model()``.
        The entire Ensemble will launch as one batch.

        Provided batch and run settings, either ``params``
        or ``replicas`` must be passed and the entire ensemble
        will launch as a single batch.

        Provided solely run settings, either ``params``
        or ``replicas`` must be passed and the Ensemble members
        will each launch sequentially.

        The kwargs argument can be used to pass custom input
        parameters to the permutation strategy.

        :param name: name of the ``Ensemble``
        :param params: parameters to expand into ``Model`` members
        :param batch_settings: describes settings for ``Ensemble`` as batch workload
        :param run_settings: describes how each ``Model`` should be executed
        :param replicas: number of replicas to create
        :param perm_strategy: strategy for expanding ``params`` into
                              ``Model`` instances from params argument
                              options are "all_perm", "step", "random"
                              or a callable function.
        :raises SmartSimError: if initialization fails
        :return: ``Ensemble`` instance
        """
        if name is None:
            raise AttributeError("Entity has no name. Please set name attribute.")
        check_path = path or osp.join(self.exp_path, name)
        entity_path: str = osp.abspath(check_path)

        try:
            new_ensemble = Ensemble(
                name=name,
                params=params or {},
                path=entity_path,
                batch_settings=batch_settings,
                run_settings=run_settings,
                perm_strat=perm_strategy,
                replicas=replicas,
                **kwargs,
            )
            return new_ensemble
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def create_model(
        self,
        name: str,
        run_settings: base.RunSettings,
        params: t.Optional[t.Dict[str, t.Any]] = None,
        path: t.Optional[str] = None,
        enable_key_prefixing: bool = False,
        batch_settings: t.Optional[base.BatchSettings] = None,
    ) -> Model:
        """Create a general purpose ``Model``

        The ``Model`` class is the most general encapsulation of
        executable code in SmartSim. ``Model`` instances are named
        references to pieces of a workflow that can be parameterized,
        and executed.

        ``Model`` instances can be launched sequentially, as a batch job,
        or as a group by adding them into an ``Ensemble``.

        All ``Models`` require a reference to run settings to specify which
        executable to launch as well provide options for how to launch
        the executable with the underlying WLM. Furthermore, batch a
        reference to a batch settings can be added to launch the ``Model``
        as a batch job through ``Experiment.start``. If a ``Model`` with
        a reference to a set of batch settings is added to a larger
        entity with its own set of batch settings (for e.g. an
        ``Ensemble``) the batch settings of the larger entity will take
        precedence and the batch setting of the ``Model`` will be
        strategically ignored.

        Parameters supplied in the `params` argument can be written into
        configuration files supplied at runtime to the ``Model`` through
        ``Model.attach_generator_files``. `params` can also be turned
        into executable arguments by calling ``Model.params_to_args``

        By default, ``Model`` instances will be executed in the
        exp_path/model_name directory if no `path` argument is supplied.
        If a ``Model`` instance is passed to ``Experiment.generate``,
        a directory within the ``Experiment`` directory will be created
        to house the input and output files from the ``Model``.

        Example initialization of a ``Model`` instance

        .. highlight:: python
        .. code-block:: python

            from smartsim import Experiment
            run_settings = exp.create_run_settings("python", "run_pytorch_model.py")
            model = exp.create_model("pytorch_model", run_settings)

            # adding parameters to a model
            run_settings = exp.create_run_settings("python", "run_pytorch_model.py")
            train_params = {
                "batch": 32,
                "epoch": 10,
                "lr": 0.001
            }
            model = exp.create_model("pytorch_model", run_settings, params=train_params)
            model.attach_generator_files(to_configure="./train.cfg")
            exp.generate(model)

        New in 0.4.0, ``Model`` instances can be colocated with an
        Orchestrator database shard through ``Model.colocate_db``. This
        will launch a single ``Orchestrator`` instance on each compute
        host used by the (possibly distributed) application. This is
        useful for performant online inference or processing
        at runtime.

        New in 0.4.2, ``Model`` instances can now be colocated with
        an Orchestrator database over either TCP or UDS using the
        ``Model.colocate_db_tcp`` or ``Model.colocate_db_uds`` method
        respectively. The original ``Model.colocate_db`` method is now
        deprecated, but remains as an alias for ``Model.colocate_db_tcp``
        for backward compatibility.

        :param name: name of the ``Model``
        :param run_settings: defines how ``Model`` should be run
        :param params: ``Model`` parameters for writing into configuration files
        :param path: path to where the ``Model`` should be executed at runtime
        :param enable_key_prefixing: If True, data sent to the ``Orchestrator``
                                     using SmartRedis from this ``Model`` will
                                     be prefixed with the ``Model`` name.
        :param batch_settings: Settings to run ``Model`` individually as a batch job.
        :raises SmartSimError: if initialization fails
        :return: the created ``Model``
        """
        if name is None:
            raise AttributeError("Entity has no name. Please set name attribute.")
        check_path = path or osp.join(self.exp_path, name)
        entity_path: str = osp.abspath(check_path)
        if params is None:
            params = {}

        try:
            new_model = Model(
                name=name,
                params=params,
                path=entity_path,
                run_settings=run_settings,
                batch_settings=batch_settings,
            )
            if enable_key_prefixing:
                new_model.enable_key_prefixing()
            return new_model
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def create_run_settings(
        self,
        exe: str,
        exe_args: t.Optional[t.List[str]] = None,
        run_command: str = "auto",
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        container: t.Optional[Container] = None,
        **kwargs: t.Any,
    ) -> settings.RunSettings:
        """Create a ``RunSettings`` instance.

        run_command="auto" will attempt to automatically
        match a run command on the system with a ``RunSettings``
        class in SmartSim. If found, the class corresponding
        to that run_command will be created and returned.

        If the local launcher is being used, auto detection will
        be turned off.

        If a recognized run command is passed, the ``RunSettings``
        instance will be a child class such as ``SrunSettings``

        If not supported by smartsim, the base ``RunSettings`` class
        will be created and returned with the specified run_command and run_args
        will be evaluated literally.

        Run Commands with implemented helper classes:
         - aprun (ALPS)
         - srun (SLURM)
         - mpirun (OpenMPI)
         - jsrun (LSF)

        :param run_command: command to run the executable
        :param exe: executable to run
        :param exe_args: arguments to pass to the executable
        :param run_args: arguments to pass to the ``run_command``
        :param env_vars: environment variables to pass to the executable
        :param container: if execution environment is containerized
        :return: the created ``RunSettings``
        """

        try:
            return settings.create_run_settings(
                self._launcher,
                exe,
                exe_args=exe_args,
                run_command=run_command,
                run_args=run_args,
                env_vars=env_vars,
                container=container,
                **kwargs,
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def create_batch_settings(
        self,
        nodes: int = 1,
        time: str = "",
        queue: str = "",
        account: str = "",
        batch_args: t.Optional[t.Dict[str, str]] = None,
        **kwargs: t.Any,
    ) -> base.BatchSettings:
        """Create a ``BatchSettings`` instance

        Batch settings parameterize batch workloads. The result of this
        function can be passed to the ``Ensemble`` initialization.

        the `batch_args` parameter can be used to pass in a dictionary
        of additional batch command arguments that aren't supported through
        the smartsim interface


        .. highlight:: python
        .. code-block:: python

            # i.e. for Slurm
            batch_args = {
                "distribution": "block"
                "exclusive": None
            }
            bs = exp.create_batch_settings(nodes=3,
                                           time="10:00:00",
                                           batch_args=batch_args)
            bs.set_account("default")

        :param nodes: number of nodes for batch job
        :param time: length of batch job
        :param queue: queue or partition (if slurm)
        :param account: user account name for batch system
        :param batch_args: additional batch arguments
        :return: a newly created BatchSettings instance
        :raises SmartSimError: if batch creation fails
        """
        try:
            return settings.create_batch_settings(
                self._launcher,
                nodes=nodes,
                time=time,
                queue=queue,
                account=account,
                batch_args=batch_args,
                **kwargs,
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    @_contextualize
    def create_database(
        self,
        port: int = 6379,
        path: t.Optional[str] = None,
        db_nodes: int = 1,
        batch: bool = False,
        hosts: t.Optional[t.Union[t.List[str], str]] = None,
        run_command: str = "auto",
        interface: str = "ipogif0",
        account: t.Optional[str] = None,
        time: t.Optional[str] = None,
        queue: t.Optional[str] = None,
        single_cmd: bool = True,
        db_identifier: str = "orchestrator",
        **kwargs: t.Any,
    ) -> Orchestrator:
        """Initialize an ``Orchestrator`` database

        The ``Orchestrator`` database is a key-value store based
        on Redis that can be launched together with other ``Experiment``
        created instances for online data storage.

        When launched, ``Orchestrator`` can be used to communicate
        data between Fortran, Python, C, and C++ applications.

        Machine Learning models in Pytorch, Tensorflow, and ONNX (i.e. scikit-learn)
        can also be stored within the ``Orchestrator`` database where they
        can be called remotely and executed on CPU or GPU where
        the database is hosted.

        To enable a SmartSim ``Model`` to communicate with the database
        the workload must utilize the SmartRedis clients. For more
        information on the database, and SmartRedis clients see the
        documentation at https://www.craylabs.org/docs/smartredis.html

        :param port: TCP/IP port
        :param db_nodes: number of database shards
        :param batch: run as a batch workload
        :param hosts: specify hosts to launch on
        :param run_command: specify launch binary or detect automatically
        :param interface: Network interface
        :param account: account to run batch on
        :param time: walltime for batch 'HH:MM:SS' format
        :param queue: queue to run the batch on
        :param single_cmd: run all shards with one (MPMD) command
        :param db_identifier: an identifier to distinguish this orchestrator in
            multiple-database experiments
        :raises SmartSimError: if detection of launcher or of run command fails
        :raises SmartSimError: if user indicated an incompatible run command
            for the launcher
        :return: Orchestrator or derived class
        """

        self._append_to_db_identifier_list(db_identifier)
        check_path = path or osp.join(self.exp_path, db_identifier)
        entity_path: str = osp.abspath(check_path)
        return Orchestrator(
            port=port,
            path=entity_path,
            db_nodes=db_nodes,
            batch=batch,
            hosts=hosts,
            run_command=run_command,
            interface=interface,
            account=account,
            time=time,
            queue=queue,
            single_cmd=single_cmd,
            launcher=self._launcher,
            db_identifier=db_identifier,
            **kwargs,
        )

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
