import sys
import time
import subprocess
from os import listdir
from os.path import isdir, basename, join

from ..state import State
from ..ensemble import Ensemble
from ..model import NumModel
from ..launcher import SlurmLauncher, LocalLauncher
from ..simModule import SmartSimModule
from ..smartSimNode import SmartSimNode
from ..error import SmartSimError, SSConfigError, SSUnsupportedError, LauncherError

from .job import Job
from .allochandler import AllocHandler
from .jobmanager import JobManager
from .dbcluster import create_cluster

from ..utils import get_logger
logger = get_logger(__name__)


class Controller(SmartSimModule):
    """The controller module provides an interface between the numerical model
       that is the subject of Smartsim and the underlying workload manager or
       run framework. There are currently three methods of execution:

          1) Slurm (implemented)

       :param State state: A State instance
       :param str launcher: The launcher type.  Accepted
                            options are 'local', and 'slurm'
    """
    def __init__(self, state, launcher=None, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Simulation Control")
        self._init_launcher(launcher)
        self._alloc_handler = AllocHandler()
        self._jobs = JobManager()

    def start(self):
        """Start the computation of a ensemble, nodes, and optionally
           facilitate communication between entities via an orchestrator.
           The fields above can be a list of strings or a list of strings
           that refer to the names of entities to be launched.

           :param str ensemble: ensemble to launch
           :param str nodes: nodes to launch
           TODO: single ensemble/model/node launch control
        """
        try:
            if self.has_orchestrator():
                if isinstance(self._launcher, LocalLauncher):
                    raise SSConfigError(
                        "Orchestrators are not supported when launching locally"
                    )

                self._prep_orchestrator()
                self._prep_nodes()
            logger.info("SmartSim State: " + self.get_state())
            self._prep_ensembles()
            self._get_allocations()
            self._launch()
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self,
             ensembles=None,
             models=None,
             nodes=None,
             stop_orchestrator=False):
        """Stops specified ensembles, nodes, and orchestrator.
           If stop_orchestrator is set to true and all ensembles and
           nodes are stopped, the orchestrato will be stopped.

           :param ensembles: List of ensembles to be stopped
           :type ensembles: list of ensemble, optional ensemble
           :param models: List of models to be stopped
           :type models: list of NumModel, option NumModel
           :param smartSimNode nodes: List of nodes to be stopped
           :type nodes: list of smartSimNode, optional smartSimNode
           :param bool stop_orchestrator: Boolean indicating if
                the ochestrator should be stopped.
        """
        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Controller.stop() is not supported when launching locally")
        self._stop_ensembles(ensembles)
        self._stop_models(models)
        self._stop_nodes(nodes)
        if stop_orchestrator:
            self._stop_orchestrator()

    def stop_all(self):
        """Stops all  ensembles, nodes, and orchestrator."""
        self.stop(ensembles=self.state.ensembles,
                  nodes=self.state.nodes,
                  stop_orchestrator=True)

    def release(self, partition=None):
        """Release the allocation(s) stopping all jobs that are currently running
           and freeing up resources. To free all resources on all partitions, invoke
           without passing a partition argument.

           :param str partition: name of the partition where the allocation is running
        """
        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Controller.release() is not supported when launching locally")
        try:
            if partition:
                try:
                    alloc_id = self._alloc_handler.allocs[partition]
                    self._launcher.free_alloc(alloc_id)
                    self._alloc_handler._remove_alloc(partition)
                except KeyError:
                    raise SmartSimError(
                        "Could not find allocation on partition: " + partition)
            else:
                allocs = self._alloc_handler.allocs.copy()
                for partition, alloc_id in allocs.items():
                    self._launcher.free_alloc(alloc_id)
                    self._alloc_handler._remove_alloc(partition)
        except SmartSimError as e:
            logger.error(e)


    def get_job(self, name):
        """Retrieve a Job instance by name. The Job object carries information about the
           job launched by the controller and it's current status.

           :param str name: name of the entity launched by the Controller
           :raises: SmartSimError
        """
        try:
            return self._jobs[name]
        except KeyError:
            raise SmartSimError("Job for " + name + " not found.")

    def get_job_nodes(self, job, wait=3):
        """Get the hostname(s) of a job from the allocation
           Wait time is necessary because Slurm take about 3 seconds to
           register that a job has been submitted.

           :param Job job: A job instance
           :param int wait: time for wait before checking nodelist after job has been launched
                            defaults to 5 seconds.
           :returns: list of hostnames given a job or dict of job_name -> nodelist
        """
        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Controller.get_job_nodes() is not supported when launching locally"
                )
        if not isinstance(job, Job):
            raise SmartSimError(
                "Argument must be a Job instance"
                )
        else:
            time.sleep(wait)
            nodes = self._get_job_nodes(job)
            return nodes

    def poll(self, interval=20, verbose=True):
        """Poll the running simulations and recieve logging
           output with the status of the job.

           :param int interval: number of seconds to wait before polling again
           :param bool verbose: set verbosity
        """
        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Controller.poll() is not supported when launching locally")

        all_finished = False
        while not all_finished:
            time.sleep(interval)
            all_finished = self.finished(verbose=verbose)

    def finished(self, verbose=True):
        """Poll all simulations and return a boolean for
           if all jobs are finished or not.

           :param bool verbose: set verbosity
           :returns: True or False for if all models have finished
        """
        # TODO make sure orchestrator doesnt effect this
        # TODO make sure NOTFOUND doesnt cause infinite loop
        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Controller.finished() is not supported when launching locally"
            )

        statuses = []
        for job in self._jobs().values():
            self._check_job(job)
            statuses.append(job.status.strip())
            if verbose:
                logger.info(job)
        if "RUNNING" in statuses:
            return False
        if "assigned" in statuses:
            return False
        if "NOTFOUND" in statuses:
            return False
        return True

    def _stop_ensembles(self, ensembles):
        """Stops specified ensembles.  If ensembles is None,
           the function returns without performning any action.
           :param ensembles: List of ensembles to be stopped
           :type ensembles: list of ensemble, optional ensemble
           :raises: SmartSimError
        """
        if ensembles == None:
            return

        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]

        if not all(isinstance(x, Ensemble) for x in ensembles):
            raise SmartSimError(
                "Only objects of type ensemble expected for variable ensembles"
            )

        for ensemble in ensembles:
            models = list(ensemble.models.values())
            self._stop_models(models)

    def _stop_models(self, models):
        """Stops the specified models.  If the models is None,
           the function returns without performing any action.

           :param models: List of models to be stopped
           :type models: list of Model, option Model
           :raises: SmartSimError
        """

        if models == None:
            return

        if isinstance(models, NumModel):
            models = [models]

        if not all(isinstance(x, NumModel) for x in models):
            raise SmartSimError(
                "Only objects of type NumModel expected for variable models")

        for model in models:
            job = self._jobs[model.name]
            self._check_job(job)
            if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
                logger.info("Stopping model " + model.name + " job " +
                            job.get_job_id())
                self._launcher.stop(job.get_job_id())
            else:
                raise SmartSimError("Unable to stop job " + job.get_job_id() +
                                    " because its status is " + job.status)

    def _stop_nodes(self, nodes):
        """Stops specified nodes.  If nodes is None,
           the function returns without performning any action.

           :param nodes: List of nodes to be stopped
           :type nodes: list of SmartSimNode, optional SmartSimNode
           :raises: SmartSimError
        """
        if nodes == None:
            return

        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]

        if not all(isinstance(x, SmartSimNode) for x in nodes):
            raise SmartSimError(
                "Only objects of type SmartSimNode expected for variable nodes"
            )

        for node in nodes:
            job = self._jobs[node.name]
            self._check_job(job)
            if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
                logger.info("Stopping node " + node.name + " job " +
                            job.get_job_id())
                self._launcher.stop(job.get_job_id())
            else:
                raise SmartSimError("Unable to stop job " + job.get_job_id() +
                                    " because its status is " + job.status)

    def _stop_orchestrator(self):
        """Stops the orchestrator only if all
           :raises: SmartSimError
        """
        for dbnode in self.state.orc.dbnodes:
            job = self._jobs[dbnode.name]
            self._check_job(job)
            if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
                logger.info("Stopping orchestrator on job " + job.get_job_id())
                self._launcher.stop(job.get_job_id())
            else:
                raise SmartSimError("Unable to stop job " + job.get_job_id() +
                                    " because its status is " + job.status)

    def _prep_nodes(self):
        """Add the nodes to the list of requirement for all the entities
           to be launched.
        """
        nodes = self.get_nodes()
        for node in nodes:
            run_settings, cmd = self._build_run_dict(node.run_settings)
            node.update_run_settings(run_settings)
            node.set_cmd(cmd)
            self._alloc_handler._add_to_allocs(run_settings)

    def _prep_orchestrator(self):
        """Add the orchestrator to the allocations requested"""
        for dbnode in self.state.orc.dbnodes:
            dbnode_settings, cmd = self._build_run_dict(dbnode.get_run_settings())
            dbnode.update_run_settings(dbnode_settings)
            dbnode.set_cmd(cmd)
            self._alloc_handler._add_to_allocs(dbnode_settings)

    def _prep_ensembles(self):
        """Add the models of each ensemble to the allocations requested"""
        ensembles = self.get_ensembles()

        for ensemble in ensembles:
            run_settings = {}
            cmd = ""
            if ensemble.name != "default":
                run_settings, cmd = self._build_run_dict(
                    ensemble.get_run_settings())
            for model in ensemble.models.values():
                if ensemble.name == "default":
                    run_settings, cmd = self._build_run_dict(
                        model.get_run_settings())
                model.update_run_settings(run_settings)
                model.set_cmd(cmd)
                self._alloc_handler._add_to_allocs(run_settings)

    def _remove_smartsim_args(self, arg_dict):
        ss_args = ["exe_args", "run_args", "executable", "run_command"]
        new_dict = dict()
        for k, v in arg_dict.items():
            if not k in ss_args:
                new_dict[k] = v
        return new_dict

    def _build_run_dict(self, entity_info):
        """Build up the run settings by retrieving the settings of the Controller
           as well as the run_settings for each entity.

           :param dict entity_info: dictionary of settings for an entity
        """
        def _build_run_command(tar_dict):
            """run_command + run_args + executable + exe_args"""

            exe = self.get_config("executable", aux=tar_dict, none_ok=False)
            exe_args = self.get_config("exe_args", aux=tar_dict, none_ok=True)
            if not exe_args:
                exe_args = ""
            cmd = " ".join((exe, exe_args))

            if isinstance(self._launcher, LocalLauncher):
                run_command = self.get_config("run_command",
                                              aux=tar_dict,
                                              none_ok=False)
                run_args = self.get_config("run_args",
                                           aux=tar_dict,
                                           none_ok=True)
                if not run_args:
                    run_args = ""
                cmd = " ".join((run_command, run_args, exe, exe_args))

            return [cmd]

        run_dict = {}
        try:
            # ensemble level values optional because there are defaults
            run_dict["nodes"] = self.get_config("nodes",
                                                aux=entity_info,
                                                none_ok=True)
            run_dict["ppn"] = self.get_config("ppn",
                                              aux=entity_info,
                                              none_ok=True)
            run_dict["duration"] = self.get_config("duration",
                                                   aux=entity_info,
                                                   none_ok=True)
            run_dict["partition"] = self.get_config("partition",
                                                    aux=entity_info,
                                                    none_ok=True)
            cmd = _build_run_command(entity_info)

            # set partition to default if none selected
            if not run_dict["partition"]:
                run_dict["partition"] = "default"

            return run_dict, cmd

        except KeyError as e:
            raise SSConfigError(
                "SmartSim could not find following required field: %s" %
                (e.args[0])) from e

    def _validate_allocations(self):
        """Validate the allocations with specific requirements provided by the user."""
        for partition, nodes in self._alloc_handler.partitions.items():
            if partition == "default":
                partition = None
            self._launcher.validate(nodes=nodes[0],
                                    ppn=nodes[1],
                                    partition=partition)

    def _get_allocations(self):
        """Validate and retrive n allocations where n is the number of partitions
           needed by the user.
        """
        if not isinstance(self._launcher, LocalLauncher):
            self._validate_allocations()

            duration = self.get_config("duration", none_ok=True)
            for partition, nodes in self._alloc_handler.partitions.items():
                launch_partition = partition
                if partition == "default":
                    launch_partition = None
                alloc_id = self._launcher.get_alloc(
                    nodes=nodes[0],
                    ppn=nodes[1],
                    partition=launch_partition,
                    duration=duration)

                self._alloc_handler.allocs[partition] = alloc_id

    def _launch(self):
        """Launch all entities within state with the configured launcher"""

        # launch orchestrator and all dbnodes
        if self.has_orchestrator():
            for dbnode in self.state.orc.dbnodes:
                self._launch_on_alloc(dbnode)
                nodes = self.get_job_nodes(self._jobs[dbnode.name])
                self.state.orc.junction.store_db_addr(nodes[0],
                                                      self.state.orc.port)

            # Create KeyDB cluster, min nodes for cluster = 3
            if len(self.state.orc.dbnodes) > 2:
                nodes = self._jobs.get_db_nodes()
                port = self.state.orc.port
                create_cluster(nodes, port)

        # launch the SmartSimNodes and update job nodes
        for node in self.get_nodes():
            self._launch_on_alloc(node)
            self.get_job_nodes(self._jobs[node.name])

        # Launch ensembles and their respective models and update job nodes
        ensembles = self.get_ensembles()
        for ensemble in ensembles:
            for model in ensemble.models.values():
                run_settings = model.get_run_settings()
                self._launch_on_alloc(model)
                self.get_job_nodes(self._jobs[model.name])

    def _launch_on_alloc(self, entity):
        """launch a SmartSimEntity on an allocation provided by a workload manager."""
        #TODO rename this

        cmd = entity.get_cmd()
        run_settings = self._remove_smartsim_args(entity.get_run_settings())

        if isinstance(self._launcher, LocalLauncher):
            pid = self._launcher.run(cmd, run_settings)
        else:
            # if orchestrator init and not a db node, setup connections to db
            if self.has_orchestrator() and entity.type != "db":
                env_vars = self.state.orc.get_connection_env_vars(entity.name)
                run_settings["env_vars"] = env_vars

            partition = run_settings["partition"]
            pid = self._launcher.run_on_alloc(
                cmd, self._alloc_handler.allocs[partition], **run_settings)

        logger.debug("Process id for " + entity.name + " is " + str(pid))
        self._jobs.add_job(entity.name, pid, entity)

    def _check_job(self, job):
        """Takes in job and sets job properties"""
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        status = self._launcher.get_sjob_stat(job_id)
        job.set_status(status)

    def _get_job_nodes(self, job):
        if job.nodes:
            return job.nodes
        else:
            job_id = job.get_job_id()
            nodes = self._launcher.get_job_nodes(job_id)
            job.nodes = nodes
            return nodes

    def _init_launcher(self, launcher):
        """Run with a specific type of launcher"""
        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher()
            # Run all ensembles locally
            elif launcher == "local":
                self._launcher = LocalLauncher()
            else:
                raise SSUnsupportedError("Launcher type not supported: " +
                                         launcher)
        else:
            raise SSConfigError(
                "Must provide a 'launcher' argument to the Controller")
