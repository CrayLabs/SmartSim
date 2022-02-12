# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

import os.path as osp
import time
from os import getcwd

from tabulate import tabulate
from tqdm import trange

from ._core import Controller, Generator, Manifest
from ._core.utils import init_default
from .database import Orchestrator
from .entity import Ensemble, Model
from .error import SmartSimError
from .log import get_logger
from .settings import settings
from .wlm import detect_launcher

logger = get_logger(__name__)


class Experiment:
    """Experiments are the Python user interface for SmartSim.

    Experiment is a factory class that creates stages of a workflow
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

    def __init__(self, name, exp_path=None, launcher="local"):
        """Initialize an Experiment instance

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

        If you wish your driver script and Experiment to be run across
        multiple system with different schedulers (workload managers)
        you can also use the `auto` argument to have the Experiment guess
        which launcher to use based on system installed binaries and libraries

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="auto")


        The Experiment path will default to the current working directory
        and if the ``Experiment.generate`` method is called, a directory
        with the Experiment name will be created to house the output
        from the Experiment.

        :param name: name for the ``Experiment``
        :type name: str
        :param exp_path: path to location of ``Experiment`` directory if generated
        :type exp_path: str, optional
        :param launcher: type of launcher being used, options are "slurm", "pbs",
                         "cobalt", "lsf", or "local". If set to "auto",
                         an attempt will be made to find an available launcher on the system.
                         Defaults to "local"
        :type launcher: str, optional
        """
        self.name = name
        if exp_path:
            if not isinstance(exp_path, str):
                raise TypeError("exp_path argument was not of type str")
            if not osp.isdir(osp.abspath(exp_path)):
                raise NotADirectoryError("Experiment path provided does not exist")
            exp_path = osp.abspath(exp_path)
        self.exp_path = init_default(osp.join(getcwd(), name), exp_path, str)

        if launcher == "auto":
            launcher = detect_launcher()

        self._control = Controller(launcher=launcher)
        self._launcher = launcher.lower()

    def start(self, *args, block=True, summary=False):
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

        Multiple instance can also be passed to the start method
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


        :param block: block execution until all non-database
                      jobs are finished, defaults to True
        :type block: bool, optional
        :param summary: print a launch summary prior to launch,
                        defaults to False
        :type summary: bool, optional
        """
        start_manifest = Manifest(*args)
        try:
            if summary:
                self._launch_summary(start_manifest)
            self._control.start(manifest=start_manifest, block=block)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, *args):
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

        :raises TypeError: if wrong type
        :raises SmartSimError: if stop request fails
        """
        try:
            stop_manifest = Manifest(*args)
            for entity in stop_manifest.models:
                self._control.stop_entity(entity)
            for entity_list in stop_manifest.all_entity_lists:
                self._control.stop_entity_list(entity_list)
        except SmartSimError as e:
            logger.error(e)
            raise

    def generate(self, *args, tag=None, overwrite=False):
        """Generate the file structure for an ``Experiment``

        ``Experiment.generate`` creates directories for each instance
        passed to organize Experiments that launch many instances

        If files or directories are attached to ``Model`` objects
        using ``Model.attach_generator_files()``, those files or
        directories will be symlinked, copied, or configured and
        written into the created directory for that instance.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the generate method.

        :param tag: tag used in `to_configure` generator files
        :type tag: str, optional
        :param overwrite: overwrite existing folders and contents,
               defaults to False
        :type overwrite: bool, optional
        """
        try:
            generator = Generator(self.exp_path, overwrite=overwrite)
            if tag:
                generator.set_tag(tag)
            generator.generate_experiment(*args)
        except SmartSimError as e:
            logger.error(e)
            raise

    def poll(self, interval=10, verbose=True):
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

        :param interval: frequency (in seconds) of logging to stdout,
                         defaults to 10 seconds
        :type interval: int, optional
        :param verbose: set verbosity, defaults to True
        :type verbose: bool, optional
        :raises SmartSimError:
        """
        try:
            self._control.poll(interval, verbose)
        except SmartSimError as e:
            logger.error(e)
            raise

    def finished(self, entity):
        """Query if a job has completed

        An instance of ``Model`` or ``Ensemble`` can be passed
        as an argument.

        Passing ``Orchestrator`` will return an error as a
        database deployment is never finished until stopped
        by the user.

        :param entity: object launched by this ``Experiment``
        :type entity: Model | Ensemble
        :returns: True if job has completed, False otherwise
        :rtype: bool
        :raises SmartSimError: if entity has not been launched
                               by this ``Experiment``
        """
        try:
            return self._control.finished(entity)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_status(self, *args):
        """Query the status of launched instances

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
            assert all([status == smartsim.status.STATUS_COMPLETED for status in statuses])

        :returns: status of the instances passed as arguments
        :rtype: list[str]
        :raises SmartSimError: if status retrieval fails
        """
        try:
            manifest = Manifest(*args)
            statuses = []
            for entity in manifest.models:
                statuses.append(self._control.get_entity_status(entity))
            for entity_list in manifest.all_entity_lists:
                statuses.extend(self._control.get_entity_list_status(entity_list))
            return statuses
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_ensemble(
        self,
        name,
        params=None,
        batch_settings=None,
        run_settings=None,
        replicas=None,
        perm_strategy="all_perm",
        **kwargs,
    ):
        """Create an ``Ensemble`` of ``Model`` instances

        Ensembles can be launched sequentially or as a batch
        if using a non-local launcher. e.g. slurm

        Ensembles require one of the following combinations
        of arguments

            - ``run_settings`` and ``params``
            - ``run_settings`` and ``replicas``
            - ``batch_settings``
            - ``batch_settings``, ``run_settings``, and ``params``
            - ``batch_settings``, ``run_settings``, and ``replicas``

        If given solely batch settings, an empty ensemble
        will be created that models can be added to manually
        through ``Ensemble.add_model()``.
        The entire ensemble will launch as one batch.

        Provided batch and run settings, either ``params``
        or ``replicas`` must be passed and the entire ensemble
        will launch as a single batch.

        Provided solely run settings, either ``params``
        or ``replicas`` must be passed and the ensemble members
        will each launch sequentially.

        The kwargs argument can be used to pass custom input
        parameters to the permutation strategy.

        :param name: name of the ensemble
        :type name: str
        :param params: parameters to expand into ``Model`` members
        :type params: dict[str, Any]
        :param batch_settings: describes settings for ``Ensemble`` as batch workload
        :type batch_settings: BatchSettings
        :param run_settings: describes how each ``Model`` should be executed
        :type run_settings: RunSettings
        :param replicas: number of replicas to create
        :type replicas: int
        :param perm_strategy: strategy for expanding ``params`` into
                              ``Model`` instances from params argument
                              options are "all_perm", "stepped", "random"
                              or a callable function. Default is "all_perm".
        :type perm_strategy: str, optional
        :raises SmartSimError: if initialization fails
        :return: ``Ensemble`` instance
        :rtype: Ensemble
        """
        try:
            new_ensemble = Ensemble(
                name,
                params,
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

    def create_model(
        self, name, run_settings, params=None, path=None, enable_key_prefixing=False
    ):
        """Create a general purpose ``Model``

        The ``Model`` class is the most general encapsulation of
        executable code in SmartSim. ``Model`` instances are named
        references to pieces of a workflow that can be parameterized,
        and executed.

        ``Model`` instances can be launched sequentially or as a batch
        by adding them into an ``Ensemble``.

        Parameters supplied in the `params` argument can be written into
        configuration files supplied at runtime to the model through
        ``Model.attach_generator_files``. `params` can also be turned
        into executable arguments by calling ``Model.params_to_args``

        By default, ``Model`` instances will be executed in the
        current working directory if no `path` argument is supplied.
        If a ``Model`` instance is passed to ``Experiment.generate``,
        a directory within the ``Experiment`` directory will be created
        to house the input and output files from the model.

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
            model = exp.create_model("pytorch_model", run_settings, params=params)
            model.attach_generator_files(to_configure="./train.cfg")
            exp.generate(model)

        New in 0.4.0, ``Model`` instances can be co-located with an
        Orchestrator database shard through ``Model.colocate_db``. This
        will launch a single ``Orchestrator`` instance on each compute
        host used by the (possibly distributed) application. This is
        useful for performant online inference or processing
        at runtime.

        :param name: name of the model
        :type name: str
        :param run_settings: defines how ``Model`` should be run,
        :type run_settings: RunSettings
        :param params: model parameters for writing into configuration files
        :type params: dict, optional
        :param path: path to where the model should be executed at runtime
        :type path: str, optional
        :param enable_key_prefixing: If True, data sent to the Orchestrator
                                     using SmartRedis from this ``Model`` will
                                     be prefixed with the ``Model`` name.
                                     Default is True.
        :type enable_key_prefixing: bool, optional
        :raises SmartSimError: if initialization fails
        :return: the created ``Model``
        :rtype: Model
        """
        path = init_default(getcwd(), path, str)
        params = init_default({}, params, dict)
        try:
            new_model = Model(name, params, path, run_settings)
            if enable_key_prefixing:
                new_model.enable_key_prefixing()
            return new_model
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_run_settings(
        self,
        exe,
        exe_args=None,
        run_command="auto",
        run_args=None,
        env_vars=None,
        **kwargs,
    ):
        """Create a ``RunSettings`` instance.

        run_command="auto" will attempt to automatically
        match a run command on the system with a RunSettings
        class in SmartSim. If found, the class corresponding
        to that run_command will be created and returned.

        if the local launcher is being used, auto detection will
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
        :type run_command: str
        :param exe: executable to run
        :type exe: str
        :param exe_args: arguments to pass to the executable
        :type exe_args: list[str], optional
        :param run_args: arguments to pass to the ``run_command``
        :type run_args: list[str], optional
        :param env_vars: environment variables to pass to the executable
        :type env_vars: dict[str, str], optional
        :return: the created ``RunSettings``
        :rtype: RunSettings
        """
        try:
            return settings.create_run_settings(
                self._launcher,
                exe,
                exe_args=exe_args,
                run_command=run_command,
                run_args=run_args,
                env_vars=env_vars,
                **kwargs,
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_batch_settings(
        self, nodes=1, time="", queue="", account="", batch_args=None, **kwargs
    ):
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

        :param nodes: number of nodes for batch job, defaults to 1
        :type nodes: int, optional
        :param time: length of batch job, defaults to ""
        :type time: str, optional
        :param queue: queue or partition (if slurm), defaults to ""
        :type queue: str, optional
        :param account: user account name for batch system, defaults to ""
        :type account: str, optional
        :param batch_args: additional batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        :return: a newly created BatchSettings instance
        :rtype: BatchSettings
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

    def create_database(
        self,
        port=6379,
        db_nodes=1,
        batch=False,
        hosts=None,
        run_command="auto",
        interface="ipogif0",
        account=None,
        time=None,
        queue=None,
        single_cmd=True,
        **kwargs,
    ):
        """Initialize an Orchestrator database

        The ``Orchestrator`` database is a key-value store based
        on Redis that can be launched together with other Experiment
        created instances for online data storage.

        When launched, ``Orchestrator`` can be used to communicate
        data between Fortran, Python, C, and C++ applications.

        Machine Learning models in Pytorch, Tensorflow, and ONNX (i.e. scikit-learn)
        can also be stored within the Orchestrator database where they
        can be called remotely and executed on CPU or GPU where
        the database is hosted.

        To enable a SmartSim ``Model`` to communicate with the database
        the workload must utilize the SmartRedis clients. For more
        information on the database, and SmartRedis clients see the
        documentation at www.craylabs.org

        :param port: TCP/IP port, defaults to 6379
        :type port: int, optional
        :param db_nodes: numver of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to False
        :type batch: bool, optional
        :param hosts: specify hosts to launch on, defaults to None
        :type hosts: list[str], optional
        :param run_command: specify launch binary or detect automatically, defaults to "auto"
        :type run_command: str, optional
        :param interface: Network interface, defaults to "ipogif0"
        :type interface: str, optional
        :param account: account to run batch on, defaults to None
        :type account: str, optional
        :param time: walltime for batch 'HH:MM:SS' format, defaults to None
        :type time: str, optional
        :param queue: queue to run the batch on, defaults to None
        :type queue: str, optional
        :param single_cmd: run all shards with one (MPMD) command, defaults to True
        :type single_cmd: bool, optional
        :raises SmartSimError: If detection of launcher or of run command fails
        :raises SmartSimError: If user indicated an incompatible run command for the launcher
        :return: Orchestrator
        :rtype: Orchestrator or derived class
        """
        return Orchestrator(
            port=port,
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
            **kwargs,
        )

    def reconnect_orchestrator(self, checkpoint):
        """Reconnect to a running ``Orchestrator``

        This method can be used to connect to a ``Orchestrator`` deployment
        that was launched by a previous ``Experiment``. This can be
        helpful in the case where separate runs of an ``Experiment``
        wish to use the same ``Orchestrator`` instance currently
        running on a system.

        :param checkpoint: the `smartsim_db.dat` file created
                           when an ``Orchestrator`` is launched
        :type checkpoint: str
        """
        try:
            orc = self._control.reload_saved_db(checkpoint)
            return orc
        except SmartSimError as e:
            logger.error(e)
            raise


    def summary(self, format="github"):
        """Return a summary of the ``Experiment``

        The summary will show each instance that has been
        launched and completed in this ``Experiment``

        :param format: the style in which the summary table is formatted,
                       for a full list of styles see:
                       https://github.com/astanin/python-tabulate#table-format,
                       defaults to "github"
        :type format: str, optional
        :return: tabulate string of ``Experiment`` history
        :rtype: str
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
                        job.history.job_times[run],
                        job.history.statuses[run],
                        job.history.returns[run],
                    ]
                )
        else:
            return tabulate(
                values, headers, showindex=True, tablefmt=format, missingval="None"
            )

    def _launch_summary(self, manifest):
        """Experiment pre-launch summary of entities that will be launched

        :param manifest: Manifest of deployables.
        :type manifest: Manifest
        """

        summary = "\n\n=== Launch Summary ===\n"
        summary += f"Experiment: {self.name}\n"
        summary += f"Experiment Path: {self.exp_path}\n"
        summary += f"Launcher: {self._launcher}\n"
        if manifest.ensembles or manifest.ray_clusters:
            summary += f"Ensembles: {len(manifest.ensembles) + len(manifest.ray_clusters)}\n"
        if manifest.models:
            summary += f"Models: {len(manifest.models)}\n"

        if self._control.orchestrator_active:
            summary += f"Database Status: active\n"
        elif manifest.db:
            summary += f"Database Status: launching\n"
        else:
            summary += f"Database Status: inactive\n"

        summary += f"\n{str(manifest)}"

        logger.info(summary)

        wait, steps = 10, 100
        prog_bar = trange(
            steps,
            desc="Launching in...",
            leave=False,
            ncols=80,
            mininterval=0.25,
            bar_format="{desc}: {bar}| {remaining} {elapsed}",
        )
        for _ in prog_bar:
            time.sleep(wait / steps)

    def __str__(self):
        return self.name
