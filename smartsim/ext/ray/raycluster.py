import os
import re
import time as _time
import uuid

from ...entity import EntityList, SmartSimEntity
from ...error import SSConfigError, SSUnsupportedError
from ...settings import AprunSettings, QsubBatchSettings, SbatchSettings, SrunSettings
from ...utils import get_logger
from ...utils.helpers import expand_exe_path, init_default

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
#  - refactor cluster -> Refactored, the nodes are all entities now, the RayCluster behaves like an EntityList
#  - refactor controller launch (rename things) -> This is now basically an enhanced EntityList launch
#  - use set_ray_address -> using set_head_log now to make launch easier
#  - refactor comments -> Should be OK now
#  - add ray_cluster utilies so there aren't so many _methods called -> STILL OPEN


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
    :param interface: Name of network interface the cluster nodes should bind to. This should be
                      set to the highest-speed network available. Common high-speed networks are
                      `ipogif0` (on Cray XC systems) and `ib0` (on InfiniBand-based systems)
    :type interface: str
    :param alloc: ID of allocation to run on, only used if launcher is Slurm and allocation is
                  obtained with ``ray.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether cluster should be launched as batch file, ignored when ``launcher`` is `local`
    :type batch: bool
    :param time: The walltime the cluster will be running for
    :type time: str
    :param password: Password to use for Redis server, which is passed as `--redis_password` to `ray start`.
                     Can be set to `auto`, which means that a password will be generated internally, to
                     a string which will be used as password, or to `None`, which means a password will not be
                     passed to `ray start`. Defaults to `auto`
    :type password: str
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
        password="auto",
    ):
        launcher = launcher.lower()
        if launcher not in ["slurm", "pbs"]:
            raise SSUnsupportedError(
                "Only the Slurm and PBS launcher are supported by RayCluster"
            )
        self._workers = workers
        self._ray_port = ray_port
        if password:
            if password == "auto":
                self._ray_password = str(uuid.uuid4())
            else:
                self._ray_password = password
        else:
            self._ray_password = None
        self._launcher = launcher
        self._run_args = run_args
        self._batch_args = batch_args
        self.ray_head = None
        self.alloc = None
        self.batch_settings = None
        self._alloc = alloc
        self._hosts = None
        self._time = time
        self._interface = interface
        self._ray_args = ray_args
        super().__init__(name=name, path=path)
        if batch:
            self._build_batch_settings()
        self.ray_head_address = None

    @property
    def batch(self):
        try:
            if self.batch_settings:
                return True
            return False
        except AttributeError:
            return False

    def _initialize_entities(self):
        self.ray_head = RayHead(
            name="ray_head",
            path=self.path,
            ray_password=self._ray_password,
            ray_port=self._ray_port,
            launcher=self._launcher,
            run_args=self._run_args,
            ray_args=self._ray_args,
            interface=self._interface,
            alloc=self._alloc,
        )

        self.entities.append(self.ray_head)

        for worker_id in range(self._workers):
            self.worker_model = RayWorker(
                name=f"ray_worker_{worker_id}",
                path=self.path,
                run_args=self._run_args,
                ray_port=self._ray_port,
                ray_password=self._ray_password,
                ray_args=self._ray_args,
                interface=self._interface,
                launcher=self._launcher,
                alloc=self._alloc,
            )
            self.entities.append(self.worker_model)

    def _build_batch_settings(self):
        if self._launcher == "pbs":
            self.batch_settings = QsubBatchSettings(
                nodes=self._workers + 1, time=self._time, batch_args=self._batch_args
            )
        elif self._launcher == "slurm":
            self.batch_settings = SbatchSettings(
                nodes=self._workers + 1, time=self._time, batch_args=self._batch_args
            )
        else:
            raise SSUnsupportedError("Only PBS and Slurm launchers are supported")

    def add_batch_args(self, batch_args):
        """Add batch arguments to Ray cluster

        :param batch_args: batch arguments to add to Ray cluster
        :type batch_args: dict[str,str]
        """
        self._batch_args.update(batch_args)

    def _parse_ray_head_node_address(self):
        """Get the ray head node host address from the log file produced
        by the head process.

        :return: address of the head host
        :rtype: str
        """
        
        head_log = os.path.join(self.ray_head.path, self.ray_head.name + ".out")

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

        self.ray_head_address = head_ip

    def get_head_address(self):
        """Return address of head node

        If address has not been initialized, returns None

        :returns: Address of head node
        :rtype: str
        """
        if not self.ray_head_address:
            self._parse_ray_head_node_address()
        return self.ray_head_address

    def get_dashboard_address(self):
        """Returns dashboard address

        The format is <head_ip>:<dashboard_port>

        :returns: Dashboard address
        :rtype: str
        """
        return self.get_head_address() + ":" + str(self.ray_head.dashboard_port)

    def _update_workers(self):
        """Update worker args before launching them."""
        for worker in range(1, len(self.entities)):
            self.entities[worker].set_head_log(f"{os.path.join(self.ray_head.path, self.ray_head.name)}.out")


def find_ray_exe():
    """Find ray executable in current path.
    """
    try:
        ray_exe = expand_exe_path("ray")
        return ray_exe
    except SSConfigError as e:
        raise SSConfigError("Could not find ray executable") from e


def find_ray_stater_script():
    """Find location of script used to start Ray nodes.
    """
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
        ray_args=None,
        launcher="slurm",
        interface="eth0",
        alloc=None,
        dash_port=8265,
    ):
        self.dashboard_port = dash_port
        self.batch_settings = None
        self.files = None

        run_args = init_default({}, run_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_port, ray_password, interface, ray_args
        )

        run_settings = self._build_run_settings(launcher, alloc, run_args, ray_exe_args)
        super().__init__(name, path, run_settings)

    def _build_ray_exe_args(self, ray_port, ray_password, interface, ray_args):

        # python script that launches ray head node
        starter_script = find_ray_stater_script()
        ray_starter_args = [
            starter_script,
            f"+port={ray_port}",
            f"+ifname={interface}",
            f"+ray-exe={find_ray_exe()}",
            f"+head",
        ]

        if ray_password:
            ray_starter_args += [f"+redis-password={ray_password}"]

        if "dashboard-port" in ray_args:
            self.dashboard_port = int(ray_args["dashboard-port"])
        ray_starter_args += [f"+dashboard-port={self.dashboard_port}"]

        used = ["block", "redis-password", "start", "head", "port", "dashboard-port"]
        extra_ray_args = []
        for key, value in ray_args.items():
            if key not in used:
                extra_ray_args += [f"+ray-args=--{key}={value}"]
        ray_starter_args += extra_ray_args

        return " ".join(ray_starter_args)

    def _build_run_settings(self, launcher, alloc, run_args, ray_exe_args):

        if launcher == "slurm":
            run_settings = self._build_srun_settings(alloc, run_args, ray_exe_args)
        elif launcher == "pbs":
            run_settings = self._build_pbs_settings(run_args, ray_exe_args)
        else:
            raise SSUnsupportedError("Only slurm, and pbs launchers are supported.")

        run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)
        return run_settings

    def _build_pbs_settings(self, run_args, ray_args):
        # TODO: explain this
        # run_args["sync-output"] = None

        # calls ray_starter.py with arguments for the ray head node
        aprun_settings = AprunSettings("python", exe_args=ray_args, run_args=run_args)
        aprun_settings.set_tasks(1)

        return aprun_settings

    def _build_srun_settings(self, alloc, run_args, ray_args):

        delete_elements(run_args, ["oversubscribe"])

        run_args["unbuffered"] = None

        srun_settings = SrunSettings(
            "python",
            exe_args=ray_args,
            run_args=run_args,
            alloc=alloc,
        )
        srun_settings.set_nodes(1)
        return srun_settings


class RayWorker(SmartSimEntity):
    def __init__(
        self,
        name,
        path,
        ray_password,
        ray_port,
        run_args=None,
        ray_args=None,
        interface="eth0",
        launcher="slurm",
        alloc=None,
    ):

        self.batch_settings = None
        self.files = None

        run_args = init_default({}, run_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_password, ray_args, ray_port, interface
        )

        run_settings = self._build_run_settings(launcher, alloc, run_args, ray_exe_args)
        super().__init__(name, path, run_settings)

    @property
    def batch(self):
        return False

    def set_head_log(self, head_log):
        """Set head log file (with full path)
        
        The head log file is used by the worker to discover
        the head IP address. This function is called by
        RayCluster before the cluster is launched.
        """
        self.run_settings.add_exe_args([f"+head-log={head_log}"])

    def _build_ray_exe_args(self, ray_password, ray_args, ray_port, interface):

        # python script that launches ray  node
        starter_script = find_ray_stater_script()
        ray_starter_args = [
            starter_script,
            f"+ray-exe={find_ray_exe()}",
            f"+port={ray_port}",
            f"+ifname={interface}",
        ]
        if ray_password:
            ray_starter_args += [f"+redis-password={ray_password}"]

        used = ["block", "redis-password", "start", "head", "port", "dashboard-port", "dashboard-host"]
        extra_ray_args = []
        for key, value in ray_args.items():
            if key not in used:
                extra_ray_args += [f"+ray-args=--{key}={value}"]
        ray_starter_args += extra_ray_args

        return " ".join(ray_starter_args)

    def _build_run_settings(self, launcher, alloc, run_args, ray_exe_args):

        if launcher == "slurm":
            run_settings = self._build_srun_settings(alloc, run_args, ray_exe_args)
        elif launcher == "pbs":
            run_settings = self._build_pbs_settings(run_args, ray_exe_args)
        else:
            raise SSUnsupportedError("Only slurm, and pbs launchers are supported.")

        run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)
        return run_settings

    def _build_pbs_settings(self, run_args, ray_args):

        # TODO: explain this
        # run_args["sync-output"] = None

        # calls ray_starter.py with arguments for a ray worker node
        aprun_settings = AprunSettings("python", exe_args=ray_args, run_args=run_args)

        return aprun_settings

    def _build_srun_settings(self, alloc, run_args, ray_args):
        delete_elements(run_args, ["oversubscribe"])
        run_args["unbuffered"] = None

        srun_settings = SrunSettings(
            "python",
            exe_args=ray_args,
            run_args=run_args,
            alloc=alloc,
        )
        srun_settings.set_nodes(1)
        return srun_settings
