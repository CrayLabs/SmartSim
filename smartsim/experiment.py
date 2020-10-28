import sys
import time
import pickle
import pandas as pd
from tqdm import trange
from pprint import pformat
from os import path, mkdir, listdir, getcwd, environ

from .control import Controller
from .generation import Generator
from .launcher import LocalLauncher
from .database import Orchestrator
from .entity import SmartSimEntity, Model, Ensemble, EntityList
from .error import SmartSimError, SSConfigError, EntityExistsError

from .utils.entityutils import seperate_entities
from .utils.helpers import colorize
from .utils import get_logger

logger = get_logger(__name__)


class Experiment:
    """In SmartSim, the Experiment class is an entity creation API
    that both houses and operates on the entities it creates.

    The Experiment interface is meant to make it quick and simple
    to get complex workflows up and running.
    """

    def __init__(self, name, exp_path=None, launcher="slurm"):
        """Initialize an Experiment

        :param name: Name of the experiment
        :type name: str
        :param launcher: type of launcher, options are "local" and "slurm",
                         defaults to "slurm"
        :type launcher: str, optional
        :param exp_path: path to where experiment will be run
        :type exp_path: str
        """
        self.name = name
        self.orc = None
        if exp_path:
            self.exp_path = exp_path
        else:
            self.exp_path = path.join(getcwd(), name)
        self._control = Controller(launcher=launcher)

    def start(self, *args, summary=False):
        """Start the SmartSim Experiment

        Start the experiment by turning all entities into jobs
        for the underlying launcher specified at experiment
        initialization. All entities in the experiment will be
        launched if arguments are not passed.
        """
        try:
            if summary:
                self._launch_summary(*args)
            self._control.start(*args)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, *args):
        """Stop specific entities launched through SmartSim.

        :raises TypeError:
        :raises SmartSimError:
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
        with the status of the job. If polling the database,
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
        :type entity: SmartSimEntity or EntityList
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
        :rtype: list of statuses for given entity or entity list
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

    def create_ensemble(
        self, name, params={}, run_settings={}, perm_strategy="all_perm", **kwargs
    ):
        """Create an Ensemble entity

        :param name: name of the ensemble
        :type name: str
        :param params: model parameters for generation strategies
        :type params: dict, optional
        :param run_settings: define how the model should be run,
        :type run_settings: dict, optional
        :raises SmartSimError: If ensemble cannot be created
        :return: the created Ensemble
        :rtype: Ensemble
        """
        try:
            new_ensemble = Ensemble(
                name,
                params,
                getcwd(),
                run_settings=run_settings,
                perm_strat=perm_strategy,
                **kwargs,
            )
            return new_ensemble
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_model(
        self, name, run_settings, params={}, path=None, enable_key_prefixing=False
    ):
        """Create a Model belonging to a specific ensemble.

        :param name: name of the model
        :type name: str
        :param run_settings: defines how the model should be run,
        :type run_settings: dict
        :param params: model parameters for generation strategies
        :type params: dict, optional
        :param path: path to where the model should be executed at runtime
        :type path: str, optional
        :param enable_key_prefixing: If true, keys sent by this model will be
                                     prefixed with the model's name.
                                     Optional, defaults to False
        :type enable_key_prefixing: bool
        :raises SmartSimError: if ensemble name provided doesn't exist
        :return: the created model
        :rtype: Model
        """
        try:
            if not path:
                path = getcwd()
            new_model = Model(name, params, path, run_settings)
            if enable_key_prefixing:
                model._key_prefixing_enabled = True
            return new_model
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_orchestrator(
        self, path=None, port=6379, overwrite=False, db_nodes=1, **kwargs
    ):
        """Create an in-memory database to run with an experiment.

        Launched entities can communicate with the orchestrator through use
        of one of the Python, C, C++ or Fortran clients.

        With the default settings, this function can be used to create
        a local orchestrator that will run in parallel with other
        entities running serially in an experiment. If launching the
        orchestrator on a machine with a workload manager, include
        "alloc" as a kwarg to launch the orchestrator on a specified
        compute resource allocation.  For creating
        clustered orchestrators accross multiple compute nodes,
        set db_nodes to 3 or larger.  Additionally, the kwarg "dpn"
        can be used to launch multiple databases per compute node.

        :param path: desired path for orchestrator output/error, defaults to cwd
        :type path: str, optional
        :param port: port orchestrator should run on, defaults to 6379
        :type port: int, optional
        :param overwrite: flag to indicate that existing orcestrator files
                          in the experiment directory should be overwritten
        :type overwrite: bool, optional
        :param db_nodes: number of database nodes in the cluster, defaults to 3
        :type db_nodes: int, optional
        :raises SmartSimError: if an orchestrator already exists
        :return: Orchestrator instance created
        :rtype: Orchestrator
        """
        try:
            if isinstance(self._control._launcher, LocalLauncher) and db_nodes > 1:
                error = "Clustered orchestrators are not supported when using the local launcher\n"
                error += (
                    "Use Experiment.create_orchestrator() for launching an orchestrator"
                )
                error += "with the local launcher"
                raise SmartSimError(error)

            if self.orc and not overwrite:
                error = "Only one orchestrator can exist within a experiment.\n"
                error += "Call with overwrite=True to replace the current orchestrator"
                raise EntityExistsError(error)

            orcpath = getcwd()
            if path:
                orcpath = path

            if db_nodes == 2:
                raise SmartSimError(
                    "Only clusters of size 1 and >= 3 are supported by Smartsim"
                )
            self.orc = Orchestrator(
                orcpath,
                port,
                db_nodes=db_nodes,
                launcher=str(self._control._launcher),
                **kwargs,
            )
            return self.orc
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
            # TODO clean this up
            if isinstance(self._control._launcher, LocalLauncher):
                raise SmartSimError(
                    "Local launcher does not support " "reconnecting to a database."
                )

            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment."
                )

            db_file = "/".join((previous_orc_dir, "smartsim_db.dat"))
            if not path.exists(db_file):
                raise SmartSimError(
                    f"The SmartSim database config file " f"{db_file} cannot be found."
                )

            try:
                with open(db_file, "rb") as pickle_file:
                    db_config = pickle.load(pickle_file)
            except (OSError, IOError) as e:
                raise SmartSimError(str(e))

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

    def get_db_address(self):
        """Return the IP address of the Orchestrator

        Get the TCP address of the orchestrator returned by pinging the
        domain name used by the workload manager e.g. nid00004 returns
        127.0.0.1

        :raises SmartSimError: if orchestrator has not been launched
        :raises SmartSimError: if database nodes cannot be found
        :returns: tcp address of orchestrator
        :rtype: list
        """
        if not self.orc:
            raise SmartSimError("No orchestrator has been initialized")
        addresses = []
        for dbnode in self.orc.entities:
            job = self._control._jobs[dbnode.name]
            if not job.nodes:
                raise SmartSimError("Database has not been launched yet.")

            for address in job.nodes:
                for port in dbnode.ports:
                    addr = ":".join((address, str(port)))
                    addresses.append(addr)
        if len(addresses) < 1:
            raise SmartSimError("Could not find nodes Database was launched on")
        return addresses

    def summary(self):
        """Return a summary of the experiment"""
        index = 0
        df = pd.DataFrame(
            columns=[
                "Name",
                "Entity-Type",
                "JobID",
                "RunID",
                "Time",
                "Status",
                "Returncode"
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
                    job.history.returns[run]
                ]
                index += 1
        return df

    def _launch_summary(self, *args):
        def sprint(p):
            print(p, flush=True)
        sprint("\n")
        models, ensembles, orchestrator = seperate_entities(args)

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
                run_settng = colorize(
                    "Ensemble Run Settings: \n" + pformat(ens.run_settings),
                    color="green",
                )
                sprint(f"{name}")
                sprint(f"{num_models}")
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
                    "Model Run Settings: \n" + pformat(model.run_settings),
                    color="green",
                )
                sprint(f"{model.name}")
                sprint(f"{parameters}")
                sprint(f"{run_settng}")
            sprint("\n")
        if orchestrator:
            sprint(colorize("=== DATABASE ===", color="cyan", bold=True))
            size = colorize(
                "# of database nodes: " + str(len(orchestrator)), color="green"
            )
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
