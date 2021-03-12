import pickle
import time
from os import getcwd
import os.path as osp
from pprint import pformat

import pandas as pd
from tqdm import trange

from .control import Controller
from .entity import Ensemble, EntityList, Model, SmartSimEntity
from .error import SmartSimError
from .generation import Generator
from .launcher import LocalLauncher
from .utils.entityutils import separate_entities
from .utils.helpers import colorize, init_default

from .utils import get_logger
logger = get_logger(__name__)


class Experiment:
    """Experiments are the main workflow tool in SmartSim.

    Experiments can create jobs to launch called ``Model``(s)
    and ``Ensemble``(s). Through the Experiment interface, users
    can programmatically create, configure, start, stop, and
    query the jobs they create.
    """

    def __init__(self, name, exp_path=None, launcher="local"):
        """Initialize an Experiment

        :param name: Name of the experiment
        :type name: str
        :param launcher: type of launcher, one of slurm, pbs, or local
        :type launcher: str, optional
        :param exp_path: path to location of Experiment directory if generated
        :type exp_path: str
        """
        self.name = name
        if exp_path:
            if not isinstance(exp_path, str):
                raise TypeError("exp_path argument was not of type str")
            if not osp.isdir(osp.abspath(exp_path)):
                raise NotADirectoryError("Experiment path provided does not exist")
            exp_path = osp.abspath(exp_path)
        self.exp_path = init_default(osp.join(getcwd(), name),
                                     exp_path, str)
        self._control = Controller(launcher=launcher)

    def start(self, *args, block=True, summary=False):
        """Start the SmartSim Experiment

        Start the experiment by turning all entities into jobs
        for the underlying launcher and launching them.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the start method.

        :param block: block execution until all non-database
                      jobs are finished, defaults to True
        :type block: bool, optional
        :param summary: print a launch summary prior to launch,
                        defaults to False
        :type summary: bool, optional
        """
        try:
            if summary:
                self._launch_summary(*args)
            self._control.start(*args, block=block)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, *args):
        """Stop specific entities launched through SmartSim.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the start method.

        :raises TypeError: if wrong entity type
        :raises SmartSimError: if stop request fails
        """
        try:
            for entity in args:
                if isinstance(entity, SmartSimEntity):
                    self._control.stop_entity(entity)
                elif isinstance(entity, EntityList):
                    self._control.stop_entity_list(entity)
                else:
                    raise TypeError(
                        f"Argument was of type {type(entity)} not SmartSimEntity or EntityList"
                    )
        except SmartSimError as e:
            logger.error(e)
            raise

    def generate(self, *args, tag=None, overwrite=False):
        """Generate the file structure for an experiment

        Generate creates directories for each entity passed
        as well as copies and writes files for each entity.

        If model objects are provided with generator files,
        those files are written into according to the parameters
        provided at model initialization.

        Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
        can all be passed as arguments to the start method.

        :param tag: tag used in `to_configure` generator files,
                    defaults to None
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

    def poll(self, interval=10, poll_db=False, verbose=True):
        """Monitor jobs through logging to stdout.

        Poll the running jobs and receive logging output
        with the status of the jobs. If polling the database,
        jobs will continue until database is manually shutdown.

        :param interval: number of seconds to wait before polling again
        :type interval: int
        :param poll_db: poll dbnodes for status as well and see
                        it in the logging output
        :type poll_db: bool
        :param verbose: set verbosity
        :type verbose: bool
        :raises SmartSimError:
        """
        try:
            self._control.poll(interval, poll_db, verbose)
        except SmartSimError as e:
            logger.error(e)
            raise

    def finished(self, entity):
        """Query if a job as completed

        :param entity: object launched by SmartSim
        :type entity: SmartSimEntity | EntityList
        :returns: True if job has completed
        :rtype: bool
        """
        try:
            return self._control.finished(entity)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_status(self, *args):
        """Query the status of an entity or entities

        :returns: status of the entity
        :rtype: list[str]
        :raises SmartSimError: if status retrieval fails
        :raises TypeError:
        """
        try:
            statuses = []
            for entity in args:
                if isinstance(entity, SmartSimEntity):
                    statuses.append(self._control.get_entity_status(entity))
                elif isinstance(entity, EntityList):
                    statuses.extend(self._control.get_entity_list_status(entity))
                else:
                    raise TypeError(
                        f"Argument was of type {type(entity)} not SmartSimEntity or EntityList"
                    )
            return statuses
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_ensemble(self,
                        name,
                        params=None,
                        batch_settings=None,
                        run_settings=None,
                        replicas=None,
                        perm_strategy="all_perm",
                        **kwargs
        ):
        """Create an ensemble of models

        if given batch settings, an empty ensemble
        will be created that models can be added to manually.
        The entire ensemble will launch as one batch.

        Provided batch and run settings, the ensemble
        will require a number of replica models to be created

        Solely provided run settings, ensembles will require
        either params or replicas to be passed

        :param name: name of the ensemble
        :type name: str
        :param params: parameters to write into attached configs
        :type params: dict[str, Any], optional
        :param batch_settings: describes settings for Ensemble as batch workload
        :type batch_settings: BatchSettings, optional
        :param run_settings: describes how each model should be executed
        :type run_settings: RunSettings, optional
        :param replicas: number of replicas to create in Ensemble
        :type replicas: int, optional
        :param perm_strategy: strategy for creating Model instances from params argument
        :type perm_strategy: str, optional
        :return: Ensemble instances
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
        """Create a Model

        :param name: name of the model
        :type name: str
        :param run_settings: defines how the model should be run,
        :type run_settings: dict
        :param params: model parameters for writing into configuration files
        :type params: dict, optional
        :param path: path to where the model should be executed at runtime
        :type path: str, optional
        :param enable_key_prefixing: If true, keys sent by this model will be
                                     prefixed with the model's name.
                                     Optional, defaults to False
        :type enable_key_prefixing: bool
        :raises SmartSimError: if Model initialization fails
        :return: the created model
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

    def reconnect_orchestrator(self, previous_orc_dir):
        """Reconnect to an orchestrator that was created in a separate
        SmartSim experiment.

        :param previous_orc_dir: Dir where previous experiment db files are located.
        :type orc_dir: str
        :raises SmartSimError: if config file is missing, or corrupted
        :return: Orchestrator instance
        :rtype: Orchestrator
        """
        try:
            if isinstance(self._control._launcher, LocalLauncher):
                raise SmartSimError(
                    "Local launcher does not support " "reconnecting to a database."
                )

            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment."
                )

            db_file = "/".join((previous_orc_dir, "smartsim_db.dat"))
            if not osp.exists(db_file):
                raise SmartSimError(
                    f"The SmartSim database config file " f"{db_file} cannot be found."
                )

            try:
                with open(db_file, "rb") as pickle_file:
                    db_config = pickle.load(pickle_file)
            except (OSError, IOError) as e:
                msg = "Could not retrieve saved database configuration"
                raise SmartSimError(msg) from e

            err_message = "The SmartSim database config file is incomplete.  "
            if not "orc" in db_config:
                raise SmartSimError(
                    err_message + "Could not find the orchestrator object."
                )

            if not db_config["orc"].port:
                raise SmartSimError(
                    err_message + "The orchestrator is missing db port " "information."
                )

            if not db_config["orc"].entities:
                raise SmartSimError(
                    err_message + "The orchestrator is missing db node " "information."
                )

            if not "db_jobs" in db_config:
                raise SmartSimError(
                    err_message + "Could not find database job objects."
                )

            for db_job in db_config["db_jobs"].values():
                self._control._jobs.db_jobs[db_job.name] = db_job

            self.orc = db_config["orc"]

            if not isinstance(self._control._launcher, LocalLauncher):
                if self.finished(self.orc):
                    raise SmartSimError(
                        "The specified database is no " "longer running"
                    )

            return self.orc

        except SmartSimError as e:
            logger.error(e)
            raise

    def summary(self):
        """Return a summary of the experiment

        :return: Dataframe of experiment history
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

    def _launch_summary(self, *args):
        """Experiment pre-launch summary of entities that will be launched"""

        def sprint(p):
            print(p, flush=True)

        sprint("\n")
        models, ensembles, orchestrator = separate_entities(args)

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
                "# of database nodes: " + str(len(orchestrator)), color="green"
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
