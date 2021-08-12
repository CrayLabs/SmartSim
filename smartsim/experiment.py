# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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
from pprint import pformat

import pandas as pd
from tqdm import trange

from smartsim.control.manifest import Manifest

from .control import Controller, Manifest
from .entity import Ensemble, Model
from .error import SmartSimError
from .generation import Generator
from .utils import get_logger
from .utils.helpers import colorize, init_default

logger = get_logger(__name__)


class Experiment:
    """Experiments are the main user interface in SmartSim.

    Experiments can create instances to launch called ``Model``
    and ``Ensemble``. Through the ``Experiment`` interface, users
    can programmatically create, configure, start, stop, poll and
    query the instances they create.
    """

    def __init__(self, name, exp_path=None, launcher="local"):
        """Example initialization

        .. highlight:: python
        .. code-block:: python

            exp = Experiment(name="my_exp", launcher="local")

        :param name: name for the ``Experiment``
        :type name: str
        :param exp_path: path to location of ``Experiment`` directory if generated
        :type exp_path: str, optional
        :param launcher: type of launcher being used, options are "slurm", "pbs",
                         "cobalt", "lsf", or "local". Defaults to "local"
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
        self._control = Controller(launcher=launcher)

    def start(self, *args, block=True, summary=False):
        """Launch instances passed as arguments

        Start the ``Experiment`` by turning specified instances into jobs
        for the underlying launcher and launching them.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the start method.
        Passing more than one ``Orchestrator`` as arguments is forbidden.

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

        :raises TypeError: if wrong type
        :raises SmartSimError: if stop request fails
        """
        try:
            stop_manifest = Manifest(*args)
            for entity in stop_manifest.models:
                self._control.stop_entity(entity)
            for entity_list in stop_manifest.ensembles:
                self._control.stop_entity_list(entity_list)
            orchestrator = stop_manifest.db
            if orchestrator:
                self._control.stop_entity_list(orchestrator)
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
        """Query the status of the specific job(s)

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to ``Experiment.get_status()``

        :returns: status of the specific job(s)
        :rtype: list[str]
        :raises SmartSimError: if status retrieval fails
        :raises TypeError:
        """
        try:
            manifest = Manifest(*args)
            statuses = []
            for entity in manifest.models:
                statuses.append(self._control.get_entity_status(entity))
            for entity_list in manifest.ensembles:
                statuses.extend(self._control.get_entity_list_status(entity_list))
            orchestrator = manifest.db
            if orchestrator:
                statuses.extend(self._control.get_entity_list_status(orchestrator))
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
        """Create a ``Model``
        By default, all ``Model`` instances start with the cwd
        as their path unless specified. Regardless of if path is
        specified, upon user passing the instance to
        ``Experiment.generate()``, the ``Model`` path will be
        overwritten and replaced with the created directory for the ``Model``

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

    def reconnect_orchestrator(self, checkpoint):
        """Reconnect to a running ``Orchestrator``

        This method can be used to connect to a Redis deployment
        that was launched by a previous ``Experiment``. This way
        users can run many experiments utilizing the same Redis
        deployment

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

    def summary(self):
        """Return a summary of the ``Experiment``

        The summary will show each instance that has been
        launched and completed in this ``Experiment``

        :return: pandas Dataframe of ``Experiment`` history
        :rtype: pd.DataFrame
        """
        index = 0
        df = pd.DataFrame(
            columns=[
                "Name",
                "Entity-Type",
                "JobID",
                "RunID",
                "Time",
                "Status",
                "Returncode",
            ]
        )
        # TODO should this include running jobs?
        for job in self._control._jobs.completed.values():
            for run in range(job.history.runs + 1):
                df.loc[index] = [
                    job.entity.name,
                    job.entity.type,
                    job.history.jids[run],
                    run,
                    job.history.job_times[run],
                    job.history.statuses[run],
                    job.history.returns[run],
                ]
                index += 1
        return df

    def _launch_summary(self, manifest):
        """Experiment pre-launch summary of entities that will be launched
        :param manifest: Manifest of deployables.
        :type manifest: Manifest
        """

        def sprint(p):
            print(p, flush=True)

        sprint("\n")
        models = manifest.models
        ensembles = manifest.ensembles
        orchestrator = manifest.db

        header = colorize("=== LAUNCH SUMMARY ===", color="cyan", bold=True)
        exname = colorize("Experiment: " + self.name, color="green", bold=True)
        expath = colorize("Experiment Path: " + self.exp_path, color="green")
        launch = colorize(
            "Launching with: " + str(self._control._launcher), color="green"
        )
        numens = colorize("# of Ensembles: " + str(len(ensembles)), color="green")
        numods = colorize("# of Models: " + str(len(models)), color="green")
        has_orc = "yes" if orchestrator else "no"
        orches = colorize("Database: " + has_orc, color="green")

        sprint(f"{header}")
        sprint(f"{exname}\n{expath}\n{launch}\n{numens}\n{numods}\n{orches}\n")

        if ensembles:
            sprint(colorize("=== ENSEMBLES ===", color="cyan", bold=True))
            for ens in ensembles:
                name = colorize(ens.name, color="green", bold=True)
                num_models = colorize(
                    "# of models in ensemble: " + str(len(ens)), color="green"
                )
                batch_settings = colorize(
                    "Batch Settings: \n" + str(ens.batch_settings),
                    color="green",
                )
                run_settng = colorize(
                    "Run Settings: \n" + str(ens.run_settings),
                    color="green",
                )
                batch = colorize(f"Launching as batch: {ens.batch}", color="green")

                sprint(f"{name}")
                sprint(f"{num_models}")
                sprint(f"{batch}")
                if ens.batch:
                    print(f"{batch_settings}")
                else:
                    sprint(f"{run_settng}")
            sprint("\n")
        if models:
            sprint(colorize("=== MODELS ===", color="cyan", bold=True))
            for model in models:
                model_name = colorize(model.name, color="green", bold=True)
                parameters = colorize(
                    "Model Parameters: \n" + pformat(model.params), color="green"
                )
                run_settng = colorize(
                    "Model Run Settings: \n" + str(model.run_settings),
                    color="green",
                )
                sprint(f"{model_name}")
                sprint(f"{parameters}")
                sprint(f"{run_settng}")
            sprint("\n")
        if orchestrator:
            sprint(colorize("=== DATABASE ===", color="cyan", bold=True))
            size = colorize(
                "# of database shards: " + str(orchestrator.num_shards), color="green"
            )
            batch = colorize(f"Launching as batch: {orchestrator.batch}", color="green")
            sprint(f"{batch}")
            sprint(f"{size}")

        sprint("\n")

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
