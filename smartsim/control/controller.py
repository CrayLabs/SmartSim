import sys
import subprocess
import time

from os import listdir
from os.path import isdir, basename, join
from ..launcher import SlurmLauncher

from ..smartSimNode import SmartSimNode
from ..target import Target
from ..error import SmartSimError, SSConfigError, SSUnsupportedError, LauncherError
from ..state import State
from ..simModule import SmartSimModule
from .job import Job
from ..model import NumModel

from ..utils import get_logger
logger = get_logger(__name__)


class AllocHandler:

    def __init__(self):
        self.partitions = {}  # partition : (node count: ppn)
        self.allocs = {} # partition : allocation_id

    def _add_to_allocs(self, run_settings):
        """Add an entities run_settings to an allocation
           Add up the total number of nodes or each partition and
           take the highest ppn value present in any of the run settings

           :param dict run_settings: dictionary of settings that include
                                     number of nodes, ppn and partition
                                     requested by the user.
        """
        # partition will be None if one is not listed
        part = run_settings["partition"]
        if not part:
            part = "default" # use default partition
        nodes = int(run_settings["nodes"])
        ppn = int(run_settings["ppn"])
        if part in self.partitions.keys():
            self.partitions[part][0] += nodes
            if self.partitions[part][1] < ppn:
                self.partitions[part][1] == ppn
        else:
            self.partitions[part] = [nodes, ppn]

    def _remove_alloc(self, partition):
        """Remove a partition from both the active allocations and the
           partitions dictionary. This is called when an allocation is
           released by the user.

           :param str partition: supplied by release. partition to be freed
        """
        self.partitions.pop(partition)
        self.allocs.pop(partition)


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
        self._jobs = []

    def start(self):
        """Start the computation of a target, nodes, and optionally
           facilitate communication between entities via an orchestrator.
           The fields above can be a list of strings or a list of strings
           that refer to the names of entities to be launched.

           :param str target: target to launch
           :param str nodes: nodes to launch
           TODO: single target/model/node launch control
        """
        try:
            if self.has_orchestrator():
                self._prep_orchestrator()
                self._prep_nodes()
            logger.info("SmartSim State: " + self.get_state())
            self._prep_targets()
            self._get_allocations()
            self._launch()
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, targets=None, models=None, nodes=None, stop_orchestrator=False):
        """Stops specified targets, nodes, and orchestrator.
           If stop_orchestrator is set to true and all targets and
           nodes are stopped, the orchestrato will be stopped.

           :param targets: List of targets to be stopped
           :type targets: list of Target, optional Target
           :param models: List of models to be stopped
           :type models: list of NumModel, option NumModel
           :param smartSimNode nodes: List of nodes to be stopped
           :type nodes: list of smartSimNode, optional smartSimNode
           :param bool stop_orchestrator: Boolean indicating if
                the ochestrator should be stopped.
        """

        if self._launcher == None:
            logger.warning("Controller.stop() is not actionable for local launchers.")
        else:
            self._stop_targets(targets)
            self._stop_models(models)
            self._stop_nodes(nodes)
            if stop_orchestrator:
                self._stop_orchestrator()

    def stop_all(self):
        """Stops all  targets, nodes, and orchestrator."""
        self.stop(targets=self.state.targets,
                  nodes=self.state.nodes,
                  stop_orchestrator=True)

    def release(self, partition=None):
        """Release the allocation(s) stopping all jobs that are currently running
           and freeing up resources. To free all resources on all partitions, invoke
           without passing a partition argument.

           :param str partition: name of the partition where the allocation is running
        """
        try:
            if partition:
                try:
                    alloc_id = self._alloc_handler.allocs[partition]
                    self._launcher.free_alloc(alloc_id)
                    self._alloc_handler._remove_alloc(partition)
                except KeyError:
                    raise SmartSimError("Could not find allocation on partition: " + partition)
            else:
                allocs = self._alloc_handler.allocs.copy()
                for partition, alloc_id in allocs.items():
                    self._launcher.free_alloc(alloc_id)
                    self._alloc_handler._remove_alloc(partition)
        except SmartSimError as e:
            logger.error(e)

    def get_jobs(self):
        """Return the list of jobs that this controller has spawned"""
        return self._jobs

    def _stop_targets(self, targets):
        """Stops specified targets.  If targets is None,
           the function returns without performning any action.
           :param targets: List of targets to be stopped
           :type targets: list of Target, optional Target
           :raises: SmartSimError
        """
        if targets == None or self._launcher == None:
            return

        if isinstance(targets, Target):
            targets = [targets]

        if not all(isinstance(x, Target) for x in targets):
            raise SmartSimError("Only objects of type Target expected for variable targets")

        for target in targets:
            models = list(target.models.values())
            self._stop_models(models)

    def _stop_models(self, models):
        """Stops the specified models.  If the models is None,
           the function returns without performing any action.

           :param models: List of models to be stopped
           :type models: list of MOdel, option Model
           :raises: SmartSimError
        """

        if models == None or self._launcher == None:
            return

        if isinstance(models, NumModel):
            models = [models]

        if not all(isinstance(x, NumModel) for x in models):
            raise SmartSimError("Only objects of type NumModel expected for variable models")

        for model in models:
            job = self.get_job(model.name)
            self._check_job(job)
            if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
                logger.info("Stopping model " + model.name + " job " + job.get_job_id())
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
        if nodes == None or self._launcher == None:
            return

        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]

        if not all(isinstance(x, SmartSimNode) for x in nodes):
            raise SmartSimError("Only objects of type SmartSimNode expected for variable nodes")

        for node in nodes:
            job = self.get_job(node.name)
            self._check_job(job)
            if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
                logger.info("Stopping node " + node.name + " job " + job.get_job_id())
                self._launcher.stop(job.get_job_id())
            else:
                raise SmartSimError("Unable to stop job " + job.get_job_id()
                                    + " because its status is " + job.status)

    def _stop_orchestrator(self):
        """Stops the orchestrator only if all
           :raises: SmartSimError
        """
        job = self.get_job('orchestrator')
        self._check_job(job)
        if not (job.status == 'NOTFOUND' or job.status == 'NAN'):
            logger.info("Stopping orchestrator on job " + job.get_job_id())
            self._launcher.stop(job.get_job_id())
        else:
            raise SmartSimError("Unable to stop job " + job.get_job_id() +
                                " because its status is " + job.status)


    def get_job(self, name):
        """Retrieve a Job instance by name. The Job object carries information about the
           job launched by the controller and it's current status.

           :param str name: name of the entity launched by the Controller
        """
        found = False
        for job in self._jobs:
            if job.obj.name == name:
                found = True
                return job
        if not found:
            raise SmartSimError("Job for " + name + " not found.")

    def get_job_nodes(self, job=None, wait=5):
        """Get the hostname(s) of a job from the allocation
           if no job listed, return a dictionary of jobs and nodelists

           :param Job job: A job instance
           :param int wait: time for wait before checking nodelist after job has been launched
                            defaults to 5 seconds.
           :returns: list of hostnames given a job or dict of job_name -> nodelist
        """
        if not job:
            node_dict = dict()
            time.sleep(wait)
            for job in self._jobs:
                node_dict[job.name] = self._get_job_nodes(job)
            return node_dict
        else:
            if not isinstance(job, Job):
                raise("Argument must be a Job instance")
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
        statuses = []
        for job in self._jobs:
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

    def _prep_nodes(self):
        nodes = self.get_nodes()
        for node in nodes:
            run_dict = self._build_run_dict(node.run_settings)
            node.update_run_settings(run_dict)
            self._alloc_handler._add_to_allocs(run_dict)

    def _prep_orchestrator(self):
        orc_settings = self.state.orc.get_run_settings()
        self._alloc_handler._add_to_allocs(orc_settings)

    def _prep_targets(self):
        targets = self.get_targets()
        if len(targets) < 1:
            raise SmartSimError("No targets to simulate!")
        for target in targets:
            run_dict = self._build_run_dict(target.get_run_settings())
            target.update_run_settings(run_dict)
            # add nodes to allocation for every model within the target
            for model in target.models.values():
                self._alloc_handler._add_to_allocs(run_dict)

    def _remove_smartsim_args(self, arg_dict):
        ss_args = ["exe_args", "run_args", "executable", "run_command"]
        for arg in ss_args:
            try:
                del arg_dict[arg]
            except KeyError:
                continue
        return arg_dict

    def _build_run_dict(self, tar_info):

        def _build_run_command(tar_dict):
            """run_command + run_args + executable + exe_args"""

            exe = self.get_config("executable", aux=tar_dict, none_ok=False)
            exe_args = self.get_config("exe_args", aux=tar_dict, none_ok=True)
            if not exe_args:
                exe_args = ""
            cmd = " ".join((exe, exe_args))

            # if using local launcher
            if not self._launcher:
                run_command = self.get_config("run_command", aux=tar_dict, none_ok=False)
                run_args = self.get_config("run_args", aux=tar_dict, none_ok=True)
                if not run_args:
                    run_args = ""
                cmd = " ".join((run_command, run_args, exe, exe_args))

            return [cmd]

        run_dict = {}
        try:
            # target level values optional because there are defaults
            run_dict["nodes"] = self.get_config("nodes", aux=tar_info, none_ok=True)
            run_dict["ppn"] = self.get_config("ppn", aux=tar_info, none_ok=True)
            run_dict["duration"] = self.get_config("duration", aux=tar_info, none_ok=True)
            run_dict["partition"] = self.get_config("partition", aux=tar_info, none_ok=True)
            run_dict["cmd"] = _build_run_command(tar_info)

            return run_dict
        except KeyError as e:
            raise SSConfigError("SmartSim could not find following required field: " +
                                e.args[0]) from e

    def _get_allocations(self):
        duration = self.get_config("duration", none_ok=True)
        try:
            for partition, nodes in self._alloc_handler.partitions.items():
                if partition == "default":
                    partition = None
                self._launcher.validate(nodes=nodes[0], ppn=nodes[1], partition=partition)
                alloc_id = self._launcher.get_alloc(nodes=nodes[0], ppn=nodes[1],
                                                    partition=partition, duration=duration)
                if partition:
                    self._alloc_handler.allocs[partition] = alloc_id
                else:
                    self._alloc_handler.allocs["default"] = alloc_id
        except LauncherError as e:
            logger.error(e)
            raise

    def _launch(self):
        # launch orchestrator and add to job ID list
        if self.has_orchestrator():
            orc_settings = self.state.orc.get_run_settings()
            orc_partition = orc_settings["partition"]
            if not orc_partition:
                orc_partition = "default"
            cmd = orc_settings.pop("cmd")
            orc_settings = self._remove_smartsim_args(orc_settings)
            orc_job_id = self._launcher.run_on_alloc(cmd,
                                                     self._alloc_handler.allocs[orc_partition],
                                                     **orc_settings)
            job = Job("orchestrator", orc_job_id, self.state.orc)
            self._jobs.append(job)
            nodes = self.get_job_nodes(job)[0] # only on one node for now

            # get and store the address of the orchestrator database
            self.state.orc.junction.store_db_addr(nodes, self.state.orc.port)
        for node in self.get_nodes():
            node_settings = node.get_run_settings()
            node_partition = node_settings["partition"]
            if not node_partition:
                node_partition = "default"
            cmd = node_settings.pop("cmd")
            node_settings = self._remove_smartsim_args(node_settings)
            env_vars = self.state.orc.get_connection_env_vars(node.name)
            node_settings["env_vars"] = env_vars
            node_job_id = self._launcher.run_on_alloc(cmd,
                                                      self._alloc_handler.allocs[node_partition],
                                                      **node_settings)
            job = Job(node.name, node_job_id, node)
            self._jobs.append(job)

        targets = self.get_targets()
        for target in targets:
            target_settings = target.get_run_settings()
            target_partition = target_settings["partition"]
            cmd = target_settings.pop("cmd")
            if not target_partition:
                target_partition = "default"
            for model in target.models.values():
                env_vars = {}
                if self.has_orchestrator():
                    env_vars = self.state.orc.get_connection_env_vars(model.name)
                target_settings["env_vars"] = env_vars
                target_settings["cwd"] = join(model.path)
                target_settings["out_file"] = join(model.path, model.name + ".out")
                target_settings["err_file"] = join(model.path, model.name + ".err")
                target_settings = self._remove_smartsim_args(target_settings)
                model_job_id = self._launcher.run_on_alloc(cmd,
                                                           self._alloc_handler.allocs[target_partition],
                                                           **target_settings)
                logger.debug("Process id for " + model.name + " is " + str(model_job_id))
                job = Job(model.name, model_job_id, model)
                self._jobs.append(job)

    def _run_with_command(self, target, run_dict):
        """Run models without a workload manager directly, instead
           using some run_command specified by the user."""
        cmd = run_dict["cmd"]
        model_dict = target.models
        for _, model in model_dict.items():
            run_model = subprocess.Popen(cmd, cwd=model.path, shell=True)
            run_model.wait()

    def _check_job(self, job):
        """Takes in job and sets job properties"""
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        status = self._launcher.get_sjob_stat(job_id)
        job.set_status(status)

    def _get_job_nodes(self, job):
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        nodes = self._launcher.get_job_nodes(job_id)
        return nodes

    def _init_launcher(self, launcher):
        """Run with a specific type of launcher"""
        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher.SlurmLauncher()
            # Run all targets locally
            elif launcher == "local":
                self._launcher = None
            else:
                raise SSUnsupportedError("Launcher type not supported: "
                                        + launcher)
        else:
            raise SSConfigError("Must provide a 'launcher' argument to the Controller")

