import os
import re
import time as _time
import uuid

from ...entity import EntityList, SmartSimEntity
from ...error import SSUnsupportedError, SSConfigError
from ...settings import (
    AprunSettings,
    QsubBatchSettings,
    SbatchSettings,
    SrunSettings,
)
from ...utils import get_logger
from ...utils.helpers import init_default, expand_exe_path

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


### TODO
#  - refactor cluster
#  - refactor controller launch (rename things)
#  - use set_ray_address
#  - refactor comments
#  - add ray_cluster utilies so there aren't so many _methods called

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
        path=os.getcwd(),
        ray_port=6789,
        ray_args={},
        workers=0,
        run_args={},
        batch_args={},
        launcher="local",
        batch=False,
        time="01:00:00",
        interface="eth0",
        alloc=None,
    ):
        launcher = launcher.lower()
        if launcher not in ["slurm", "pbs"]:
            raise SSUnsupportedError(
                "Only the Slurm and PBS launcher are supported by RayCluster"
            )
        self._workers = workers
        self._ray_port = ray_port
        self._ray_password = str(uuid.uuid4())
        self._launcher = launcher
        self._run_args = run_args
        self._batch_args = batch_args
        self.ray_head = None
        self.worker_model = None
        self.alloc = None
        self.batch_settings = None
        self._alloc = alloc
        self._batch = batch
        self._hosts = None
        self._time = time
        self._interface = interface
        self._ray_args = ray_args
        super().__init__(name=name, path=path)

    @property
    def batch(self):
        return self._batch

    def _update_worker_model(self):
        self.worker_model.update_run_settings()

    def _initialize_entities(self):
        self.ray_head = RayHead(
            name="ray_head",
            params=None,
            path=self.path,
            ray_password=self._ray_password,
            ray_port=self._ray_port,
            launcher=self._launcher,
            run_args=self._run_args,
            batch_args=self._batch_args,
            ray_args=self._ray_args,
            interface = self._interface,
            alloc=self._alloc,
            batch=self.batch,
            time=self._time,
        )

        self.entities.append(self.ray_head)

        if self._workers > 0:
            self.worker_model = RayWorker(
                name="ray_worker",
                params=None,
                path=self.path,
                run_args=self._run_args,
                batch_args=self._batch_args,
                workers=self._workers,
                ray_port=self._ray_port,
                ray_password=self._ray_password,
                ray_args=self._ray_args,
                ray_head=self.ray_head,
                interface=self._interface,
                launcher=self._launcher,
                alloc=self._alloc,
                batch=self.batch,
                time=self._time,
            )
            # Insert at first position, because we want this top be stopped first
            self.entities.insert(0, self.worker_model)

    def add_batch_args(self, batch_args):
        """Add batch arguments to Ray head and workers

        :param batch_args: batch arguments to add to Ray head and workers
        :type batch_args: dict[str,str]
        """
        for ray_entity in self.entities:
            ray_entity._batch_args.update(batch_args)

    def _parse_ray_head_node_address(self):
        """Get the ray head node host address from the log file produced
        by the head process.

        :return: address of the head host
        :rtype: str
        """
        # We can rely on the file name, because we set it when we create
        # the head model
        head_log = os.path.join(self.ray_head.path, "ray_head.out")

        max_attempts = 60
        attempts = 0
        while not os.path.isfile(head_log):
            _time.sleep(1)
            attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")

        attempts = 0
        head_ip = None
        while head_ip is None:
            _time.sleep(1)
            with open(head_log) as fp:
                line = fp.readline()
                while line:
                    plain_line = re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", line)
                    if "Local node IP:" in plain_line:
                        matches = re.search(r"(?<=Local node IP: ).*", plain_line)
                        head_ip = matches.group()
                        break
                    line = fp.readline()
            attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Could not find Ray cluster head address.")

        self.ray_head.address = head_ip

    def get_head_address(self):
        """Return address of head node

        If address has not been initialized, returns None

        :returns: Address of head node
        :rtype: str
        """
        return self.ray_head.address

    def get_dashboard_address(self):
        """Returns address of dashboard address

        The format is <head_ip>:<dashboard_port>

        :returns: Dashboard address
        :rtype: str
        """
        return self.ray_head.address + ":" + str(self.ray_head.dashboard_port)




def find_ray_exe():
    try:
        ray_exe = expand_exe_path("ray")
        return ray_exe
    except SSConfigError as e:
        raise SSConfigError(
            "Could not find ray executable") from e


def find_ray_stater_script():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return f"{dir_path}/raystarter.py"


class RayHead(SmartSimEntity):

    def __init__(
        self,
        name,
        path,
        ray_password,
        ray_port=6789,
        run_args=None,
        batch=False,
        batch_args=None,
        ray_args=None,
        launcher="slurm",
        interface="eth0",
        alloc=None,
        time="01:00:00",
        dash_port=8265
        ):
        self.dashboard_port = dash_port
        self.batch_settings = None

        run_args = init_default({}, run_args, dict)
        batch_args = init_default({}, batch_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_port,
            ray_password,
            interface,
            ray_args
        )

        run_settings = self._build_run_settings(
            launcher,
            batch,
            alloc,
            time,
            run_args,
            batch_args,
            ray_exe_args
        )
        super().__init__(name, path, run_settings)

    @property
    def batch(self):
        try:
            if self.batch_settings:
                return True
            return False
        except AttributeError:
            return False

    def _build_ray_exe_args(self, ray_port, ray_password, interface, ray_args):

        # python script that launches ray head node
        starter_script = find_ray_stater_script()
        ray_starter_args = [
            starter_script,
            f"--port={ray_port}",
            f"--redis-password={ray_password}",
            f"--ifname={interface}",
            f"--ray-exe={find_ray_exe()}"
        ]

        if "dashboard-port" in ray_args:
            self.dashboard_port = int(ray_args["dashboard-port"])
        ray_starter_args += [f"--dashboard-port={self.dashboard_port}"]

        used = ["block", "redis-password", "start", "head", "port", "dashboard-port"]
        extra_ray_args = []
        for key, value in ray_args.items():
            if key not in used:
                extra_ray_args += [f"--{key}={value}"]
        ray_starter_args += [f'--ray-args="{" ".join(extra_ray_args)}"']

        return " ".join(ray_starter_args)


    def _build_run_settings(self,
                            launcher,
                            batch,
                            alloc,
                            time,
                            run_args,
                            batch_args,
                            ray_exe_args):

        if launcher == "slurm":
            run_settings = self._build_srun_settings(
                batch,
                alloc,
                run_args,
                batch_args,
                time,
                ray_exe_args
            )
        elif launcher == "pbs":
            run_settings = self._build_pbs_settings(
                batch,
                run_args,
                batch_args,
                time,
                ray_exe_args
            )
        else:
            raise SSUnsupportedError(
            "Only slurm, and pbs launchers are supported."
        )

        run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)
        return run_settings


    def _build_pbs_settings(self, batch, run_args, batch_args, time, ray_args):
        if batch:
            self.batch_settings = QsubBatchSettings(
                nodes=1,
                ncpus=1,
                time=time,
                batch_args=batch_args
            )

        # TODO: explain this
        run_args["sync-output"] = None

        # calls ray_starter.py with arguments for the ray head node
        aprun_settings = AprunSettings(
            "python",
            exe_args=ray_args,
            run_args=run_args
        )
        aprun_settings.set_tasks(1)

        return aprun_settings

    def _build_srun_settings(self, batch, alloc, run_args, batch_args, time, ray_args):
        delete_elements(batch_args, ["nodes", "ntasks-per-node"])

        if batch:
            self.batch_settings = SbatchSettings(
                nodes=1,
                time=time,
                batch_args=batch_args
            )

        delete_elements(run_args, ["oversubscribe"])

        run_args["unbuffered"] = None
        if not batch and not alloc:
            run_args["overcommit"] = None

        srun_settings = SrunSettings(
            "python",
            exe_args=ray_args,
            run_args=run_args,
            alloc=alloc,
        )
        srun_settings.set_nodes(1)
        return srun_settings


class RayWorkerSet(SmartSimEntity):

    def __init__(
        self,
        name,
        path,
        workers,
        ray_password,
        run_args=None,
        batch=False,
        batch_args=None,
        ray_args=None,
        launcher="slurm",
        alloc=None,
        time="01:00:00",
        ):

        self.workers = workers
        self.batch_settings = None

        run_args = init_default({}, run_args, dict)
        batch_args = init_default({}, batch_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_password,
            ray_args
        )

        run_settings = self._build_run_settings(
            launcher,
            batch,
            alloc,
            time,
            run_args,
            batch_args,
            ray_exe_args
        )
        super().__init__(name, path, run_settings)

    @property
    def batch(self):
        try:
            if self.batch_settings:
                return True
            return False
        except AttributeError:
            return False

    def set_ray_address(self, address):
        self.run_settings.add_exe_args([f"--address={address}"])

    def _build_ray_exe_args(self, ray_password, ray_args):

        ray_exe_args = [
            "start",
            f"--redis-password={ray_password}",
            "--block"
        ]

        used = ["block", "redis-password", "start", "head"]
        for key, value in ray_args.items():
            if key not in used:
                ray_exe_args += [f"--{key}={value}"]

        ray_exe_args = " ".join(ray_exe_args)
        return ray_exe_args


    def _build_run_settings(self,
                            launcher,
                            batch,
                            alloc,
                            time,
                            run_args,
                            batch_args,
                            ray_exe_args):

        if launcher == "slurm":
            run_settings = self._build_srun_settings(
                batch,
                alloc,
                run_args,
                batch_args,
                time,
                ray_exe_args
            )
        elif launcher == "pbs":
            run_settings = self._build_pbs_settings(
                batch,
                run_args,
                batch_args,
                time,
                ray_exe_args
            )
        else:
            raise SSUnsupportedError(
            "Only slurm, and pbs launchers are supported."
        )

        # as many tasks as workers
        run_settings.set_tasks(self.workers)
        run_settings.set_tasks_per_node(1)
        return run_settings


    def _build_pbs_settings(self, batch, run_args, batch_args, time, ray_args):
        if batch:
            self.batch_settings = QsubBatchSettings(
                nodes=self.workers,
                ncpus=1,
                time=time,
                batch_args=batch_args
            )

        # TODO: explain this
        run_args["sync-output"] = None

        # calls ray_starter.py with arguments for the ray head node
        aprun_settings = AprunSettings(
            "ray",
            exe_args=ray_args,
            run_args=run_args
        )

        return aprun_settings

    def _build_srun_settings(self, batch, alloc, run_args, batch_args, time, ray_args):
        delete_elements(batch_args, ["nodes", "ntasks-per-node"])

        if batch:
            self.batch_settings = SbatchSettings(
                nodes=self.workers,
                time=time,
                batch_args=batch_args
            )

        delete_elements(run_args, ["oversubscribe"])

        run_args["unbuffered"] = None
        if not batch and not alloc:
            run_args["overcommit"] = None

        srun_settings = SrunSettings(
            "ray",
            exe_args=ray_args,
            run_args=run_args,
            alloc=alloc,
        )
        srun_settings.set_nodes(self.workers)
        return srun_settings

