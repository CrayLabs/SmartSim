import os
import re
import shlex
import time
import uuid

from ...entity import EntityList, Model
from ...error import SSUnsupportedError
from ...settings import (
    AprunSettings,
    QsubBatchSettings,
    RunSettings,
    SbatchSettings,
    SrunSettings,
)
from ...utils import get_logger

logger = get_logger(__name__)


def delete_elements(dictionary, key_list):
    """Delete elements from a dictionary.
    :param dictionary: the dictionary from which the elements must be deleted.
    :type dictionary: dict
    :param key_list: the list of keys to delete from the dictionary.
    :type key: any
    """
    for key in key_list:
        if key in dictionary:
            del dictionary[key]


class RayCluster(EntityList):
    """Entity used to run a Ray cluster on a given number of hosts. Ray is launched on each host,
    and the first host is used to launch the head node.

    :param name: The name of the entity.
    :type name: str
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_port: Port at which the head node will be running.
    :type ray_port: int
    :param ray_args: Arguments to be passed to Ray executable. Each dictionary entry will be added
                     to the Ray command line as `--key=value`, or `--key` if `value` is set to `None`.
    :type ray_args: dict[str,str]
    :param workers: Number of workers (the total size of the cluster is ``workers``+1).
    :type workers: int
    :param run_args: Arguments to pass to launcher to specify details such as partition, time, and so on.
    :type run_args: dict[str,str]
    :param batch_args: Additional batch arguments passed to launcher when running batch jobs.
    :type batch_args: dict[str,str]
    :param launcher: Name of launcher to use for starting the cluster.
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether cluster should be launched as batch file, ignored when ``launcher`` is `local`
    :type batch: bool
    :param time: The walltime the cluster will be running for
    :type time: str
    """

    def __init__(
        self,
        name,
        path=".",
        ray_port=6780,
        ray_args={},
        workers=0,
        run_args={},
        batch_args={},
        launcher="local",
        batch=False,
        time="01:00:00",
        alloc=None,
    ):
        if launcher == "local" and workers > 0:
            raise SSUnsupportedError(
                "Cannot launch a local Ray cluster with more than one node."
                "Set workers to 0 and restart."
            )
        self._workers = workers
        self._ray_port = ray_port
        self._ray_password = str(uuid.uuid4())
        self._launcher = launcher.lower()
        self._run_args = run_args
        self._batch_args = batch_args
        self.head_model = None
        self.worker_model = None
        self.alloc = None
        self.batch_settings = None
        self._alloc = alloc
        self._batch = launcher != "local" and batch
        self._hosts = None
        self._time = time
        self._ray_args = ray_args
        super().__init__(name=name, path=path)

    @property
    def batch(self):
        return self._batch

    def _update_worker_model(self):
        self.worker_model.update_run_settings()

    def _initialize_entities(self):
        self.head_model = RayHead(
            name="head",
            params=None,
            path=os.path.join(self.path, "head"),
            ray_password=self._ray_password,
            ray_port=self._ray_port,
            launcher=self._launcher,
            run_args=self._run_args,
            batch_args=self._batch_args,
            ray_args=self._ray_args,
            alloc=self._alloc,
            batch=self.batch,
            time=self._time,
        )

        self.entities.append(self.head_model)

        if self._workers > 0:
            self.worker_model = RayWorker(
                name="workers",
                params=None,
                path=os.path.join(self.path, "workers"),
                run_args=self._run_args,
                batch_args=self._batch_args,
                workers=self._workers,
                ray_port=self._ray_port,
                ray_password=self._ray_password,
                ray_args=self._ray_args,
                head_model=self.head_model,
                launcher=self._launcher,
                alloc=self._alloc,
                batch=self.batch,
                time=self._time,
            )
            # Insert at first position, because we want this top be stopped first
            self.entities.insert(0, self.worker_model)

    def add_batch_args(self, batch_args):
        """Add batch argumentss to all Ray nodes

        :param batch_args: batch arguments to add to Ray nodes
        :type batch_args: dict[str,str]
        """
        for node in self.entities:
            node._batch_args.update(batch_args)

    def _parse_ray_head_node_address(self):
        """Get the ray head node host address from the log file produced
        by the head process.

        :return: address of the head host
        :rtype: str
        """
        # We can rely on the file name, because we set it when we create
        # the head model
        head_log = os.path.join(self.head_model.path, "head.out")

        max_attempts = 60
        attempts = 0
        while not os.path.isfile(head_log):
            time.sleep(1)
            attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")

        attempts = 0
        head_ip = None
        while head_ip is None:
            time.sleep(1)
            with open(head_log) as fp:
                line = fp.readline()
                while line:
                    plain_line = re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", line)
                    if "Local node IP:" in plain_line:
                        matches = re.search(r"(?<=Local node IP: ).*", plain_line)
                        head_ip = matches.group()
                    line = fp.readline()
            attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")

        self.head_model.address = head_ip

    def get_head_address(self):
        """Return address of head node

        If address has not been initialized, returns None

        :returns: Address of head node
        :rtype: str
        """
        return self.head_model.address

    def get_dashboard_address(self):
        """Returns address of dashboard address

        The format is <head_ip>:<dashboard_port>

        :returns: Dashboard address
        :rtype: str
        """
        return self.head_model.address + ":" + self.head_model.dashboard_port


class RayHead(Model):
    """Ray head node model.

    :param name: The name of the entity.
    :type name: str
    :param params: model parameters for writing into configuration files.
    :type params: dict
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_password: Password used to connect to Redis by Ray
    :type ray_password: str
    :param ray_port: Port at which the head node will be running
    :type ray_port: int
    :param ray_args: Arguments to be passed to Ray executable. Each dictionary entry will be added
                     to the Ray command line as `--key=value`, or `--key` if `value` is set to `None`.
    :type ray_args: dict[str,str]
    :param batch_args: Additional batch arguments passed to launcher when running batch jobs.
    :type batch_args: dict[str,str]
    :param launcher: Name of launcher to use for starting the cluster
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``smartsim.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether the head node should be launched through a batch file
    :type batch: bool
    :param time: The walltime the head node will be running for
    :type time: str
    """

    def __init__(
        self,
        name,
        params,
        path,
        ray_password,
        ray_port=6780,
        run_args={},
        batch_args={},
        ray_args={},
        launcher="local",
        alloc=None,
        batch=False,
        time="01:00:00",
    ):
        self._ray_port = ray_port
        self._run_args = run_args.copy()
        self._batch_args = batch_args.copy()
        self._ray_password = ray_password
        self._ray_args = ray_args
        self._alloc = alloc
        self._launcher = launcher
        self.address = None
        self._batch = launcher != "local" and batch
        self._time = time
        self._hosts = []
        self.dashboard_port = 8265

        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)

    @property
    def batch(self):
        return self._batch

    def _build_run_settings(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ray_starter_args = [f"{dir_path}/raystarter.py"]
        ray_starter_args += [f"--port={self._ray_port}"]
        ray_starter_args += [f"--redis-password={self._ray_password}"]
        if "dashboard-port" in self._ray_args:
            self.dashboard_port = int(self._ray_args["dashboard-port"])
        ray_starter_args += [f"--dashboard-port={self.dashboard_port}"]

        delete_elements(
            self._ray_args,
            ["block", "redis-password", "start", "head", "port", "dashboard-port"],
        )

        ray_args = []
        for key in self._ray_args.keys():
            ray_args += [f"--{key}={self._ray_args[key]}"]

        ray_starter_args_str = " ".join(
            ray_starter_args + [f'--ray-args="{" ".join(ray_args)}"']
        )

        if self._launcher == "slurm":
            self.run_settings = self._build_srun_settings(ray_starter_args_str)
        elif self._launcher == "pbs":
            self.run_settings = self._build_pbs_settings(ray_starter_args_str)
        elif self._launcher == "local":
            self.run_settings = RunSettings(
                "python", ray_starter_args_str, run_args=self._run_args
            )
        else:
            raise NotImplementedError(
                "Only Slurm, local, and PBS launchers are supported."
            )

        if self._launcher != "local":
            self.run_settings.set_walltime(self._time)
            self.run_settings.set_tasks(1)
            self.run_settings.set_tasks_per_node(1)

    def _build_pbs_settings(self, ray_args):
        if self.batch:
            self.batch_settings = QsubBatchSettings(
                nodes=1, ncpus=1, time=self._time, batch_args=self._batch_args
            )
        self._run_args["sync-output"] = None

        aprun_settings = AprunSettings(
            "python", exe_args=ray_args, run_args=self._run_args
        )
        aprun_settings.set_tasks(1)

        return aprun_settings

    def _build_srun_settings(self, ray_args):
        delete_elements(self._batch_args, ["nodes", "ntasks-per-node"])

        if self.batch:
            self.batch_settings = SbatchSettings(
                nodes=1, time=self._time, batch_args=self._batch_args
            )

        delete_elements(self._run_args, ["oversubscribe"])

        self._run_args["unbuffered"] = None
        if not self.batch and not self._alloc:
            self._run_args["overcommit"] = None
        srun_settings = SrunSettings(
            "python",
            exe_args=ray_args,
            run_args=self._run_args,
            expand_exe=False,
            alloc=self._alloc,
        )
        srun_settings.set_nodes(1)
        return srun_settings


class RayWorker(Model):
    """Ray head node model.

    :param name: The name of the entity.
    :type name: str
    :param params: model parameters for writing into configuration files.
    :type params: dict
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_password: Password used to connect to Redis by Ray
    :type ray_password: str
    :param ray_port: Port at which the head node will be running
    :type ray_port: int
    :param ray_args: Arguments to be passed to Ray executable. Each dictionary entry will be added
                     to the Ray command line as `--key=value`, or `--key` if `value` is set to `None`.
    :type ray_args: dict[str,str]
    :param batch_args: Additional batch arguments passed to launcher when running batch jobs.
    :type batch_args: dict[str,str]
    :param head_model: This cluster's head model's entity
    :type head_model: RayHead
    :param launcher: Name of launcher to use for starting the cluster
    :type launcher: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether the head node should be launched through a batch file
    :type batch: bool
    :param time: The walltime the head node will be running for
    :type time: str
    """

    def __init__(
        self,
        name,
        params,
        path,
        workers,
        head_model,
        ray_password,
        ray_port=6780,
        ray_args={},
        run_args={},
        batch_args={},
        launcher="local",
        alloc=None,
        batch=False,
        time="01:00:00",
    ):
        self._run_args = run_args.copy()
        self._batch_args = batch_args.copy()
        self._ray_port = ray_port
        self._ray_password = ray_password
        self._ray_args = ray_args
        self._launcher = launcher
        self._workers = workers
        self.head_model = head_model
        self._alloc = alloc
        self._batch = batch
        self._time = time

        self._build_run_settings()
        super().__init__(name, params, path, self.run_settings)

    @property
    def batch(self):
        return self._batch

    def update_run_settings(self):
        self.run_settings.add_exe_args(
            [f"--address={self.head_model.address}:{self._ray_port}"]
        )

    def _build_run_settings(self):
        ray_args = ["start"]
        ray_args += [f"--redis-password={self._ray_password}"]
        ray_args += ["--block"]
        delete_elements(self._ray_args, ["block", "redis-password", "start", "head"])

        for key in self._ray_args.keys():
            ray_args += [f"--{key}={self._ray_args[key]}"]

        ray_args_str = " ".join(ray_args)

        if self._launcher == "slurm":
            self.run_settings = self._build_srun_settings(ray_args_str)
        elif self._launcher == "pbs":
            self.run_settings = self._build_pbs_settings(ray_args_str)
        elif self._launcher == "local":
            raise SSUnsupportedError("Ray workers cannot be launched locally.")
        else:
            raise NotImplementedError("Only Slurm and PBS launchers are supported.")

        if self._time:
            self.run_settings.set_walltime(self._time)
        self.run_settings.set_tasks(self._workers)
        self.run_settings.set_tasks_per_node(1)

    def _build_srun_settings(self, ray_args):
        delete_elements(self._batch_args, ["nodes", "ntasks-per-node"])

        if self.batch:
            self.batch_settings = SbatchSettings(
                nodes=self._workers, time=self._time, batch_args=self._batch_args
            )

        self._run_args["unbuffered"] = None
        if not self.batch and self._alloc is None:
            self._run_args["overcommit"] = None

        srun_settings = SrunSettings(
            "ray",
            exe_args=ray_args,
            run_args=self._run_args,
            expand_exe=False,
            alloc=self._alloc,
        )
        srun_settings.set_nodes(self._workers)
        srun_settings.set_tasks_per_node(1)
        return srun_settings

    def _build_pbs_settings(self, ray_args):
        if self.batch:
            self.batch_settings = QsubBatchSettings(
                nodes=self._workers,
                batch_args=self._batch_args,
                ncpus=1,
                time=self._time,
            )
        self._run_args["sync-output"] = None
        aprun_settings = AprunSettings(
            "ray", exe_args=ray_args, run_args=self._run_args, expand_exe=False
        )
        return aprun_settings
