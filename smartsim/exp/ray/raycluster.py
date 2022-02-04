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

import os
import re
import time as _time
import uuid
import sys

from ..._core.utils import init_default
from ..._core.utils.helpers import expand_exe_path
from ...entity import EntityList, SmartSimEntity
from ...error import SmartSimError, SSUnsupportedError
from ...log import get_logger
from ...settings import settings

logger = get_logger(__name__)


class RayCluster(EntityList):
    """Entity used to run a Ray cluster on a given number of hosts. One Ray node is
    launched on each host, and the first host is used to launch the head node.

    :param name: The name of the entity.
    :type name: str
    :param path: path to output, error, and configuration files
    :type path: str
    :param ray_port: Port at which the head node will be running.
    :type ray_port: int
    :param ray_args: Arguments to be passed to Ray executable.
    :type ray_args: dict[str,str]
    :param num_nodes: Number of hosts, includes 1 head node and all worker nodes.
    :type num_nodes: int
    :param run_args: Arguments to pass to launcher to specify details such as partition or time.
    :type run_args: dict[str,str]
    :param batch_args: Additional batch arguments passed to launcher when running batch jobs.
    :type batch_args: dict[str,str]
    :param launcher: Name of launcher to use for starting the cluster.
    :type launcher: str
    :param interface: Name of network interface the cluster nodes should bind to.
    :type interface: str
    :param alloc: ID of allocation to run on, if obtained with ``smartsim.slurm.get_allocation``
    :type alloc: int
    :param batch: Whether cluster should be launched as batch file, ignored when ``launcher`` is `local`
    :type batch: bool
    :param time: The walltime the cluster will be running for
    :type time: str
    :param run_command: specify launch binary, defaults to automatic selection.
    :type run_command: str
    :param hosts: specify hosts to launch on, defaults to None. Optional if not launching with OpenMPI.
    :type hosts: str, list[str]
    :param password: Password to use for Redis server, which is passed as `--redis_password` to `ray start`.
                     Can be set to
                     - `auto`: a strong password will be generated internally
                     - a string: it will be used as password
                     - `None`: the default Ray password will be used.
                     Defaults to `auto`
    :type password: str
    """

    def __init__(
        self,
        name,
        path=os.getcwd(),
        ray_port=6789,
        ray_args=None,
        num_nodes=1,
        run_args=None,
        batch_args=None,
        launcher="local",
        batch=False,
        time="01:00:00",
        interface="ipogif0",
        alloc=None,
        run_command=None,
        host_list=None,
        password="auto",
        **kwargs,
    ):
        launcher = launcher.lower()
        supported_launchers = ["slurm", "pbs", "cobalt"]
        if launcher not in supported_launchers:
            raise SSUnsupportedError(
                "The supported launchers for RayCluster are",
                *[f"{launcher_name}," for launcher_name in supported_launchers],
                f"but {launcher} was provided.",
            )

        if password:
            if password == "auto":
                self._ray_password = str(uuid.uuid4())
            else:
                self._ray_password = password
        else:
            self._ray_password = None

        if num_nodes < 1:
            raise ValueError("Number of nodes must be larger than 0.")

        self.alloc = None
        self.batch_settings = None
        self._hosts = None

        run_args = init_default({}, run_args, dict)
        batch_args = init_default({}, batch_args, dict)
        ray_args = init_default({}, ray_args, dict)

        self._ray_args = ray_args
        super().__init__(
            name=name,
            path=path,
            ray_args=ray_args,
            run_args=run_args,
            ray_port=ray_port,
            launcher=launcher,
            interface=interface,
            alloc=alloc,
            num_nodes=num_nodes,
            run_command=run_command if run_command else "auto",
            host_list=host_list,
            **kwargs,
        )
        if batch:
            self.batch_settings = settings.create_batch_settings(
                launcher=launcher,
                nodes=num_nodes,
                time=time,
                batch_args=batch_args,
                **kwargs,
            )
        self.ray_head_address = None

        if host_list:
            self.set_hosts(host_list=host_list, launcher=launcher)

    @property
    def batch(self):
        try:
            if self.batch_settings:
                return True
            return False
        except AttributeError:
            return False

    def set_hosts(self, host_list, launcher):
        """Specify the hosts for the ``RayCluster`` to launch on. This is
        optional, unless ``run_command`` is `mpirun`.

        :param host_list: list of hosts (compute node names)
        :type host_list: str | list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        # TODO check length
        if self.batch:
            self.batch_settings.set_hostlist(host_list)
        for host, node in zip(host_list, self.entities):
            # Aprun doesn't like settings hosts in batch launch
            if launcher == "pbs" or launcher == "cobalt":
                if not self.batch:
                    node.run_settings.set_hostlist([host])
            else:
                node.run_settings.set_hostlist([host])

    def _initialize_entities(self, **kwargs):

        ray_port = kwargs.get("ray_port", 6789)
        launcher = kwargs.get("launcher", "slurm")
        ray_args = kwargs.get("ray_args", None)
        run_args = kwargs.get("run_args", None)
        interface = kwargs.get("interface", "ipogif0")
        num_nodes = kwargs.get("num_nodes", 0)
        alloc = kwargs.get("alloc", None)
        run_command = kwargs.get("run_command", None)

        ray_head = RayHead(
            name="ray_head",
            path=self.path,
            ray_password=self._ray_password,
            ray_port=ray_port,
            launcher=launcher,
            run_args=run_args.copy(),
            ray_args=ray_args.copy(),
            interface=interface,
            run_command=run_command,
            alloc=alloc,
        )

        self.entities.append(ray_head)

        for worker_id in range(num_nodes - 1):
            worker_model = RayWorker(
                name=f"ray_worker_{worker_id}",
                path=self.path,
                run_args=run_args.copy(),
                ray_port=ray_port,
                ray_password=self._ray_password,
                ray_args=ray_args.copy(),
                interface=interface,
                run_command=run_command,
                launcher=launcher,
                alloc=alloc,
            )
            self.entities.append(worker_model)

    def get_head_address(self):
        """Return address of head node

        If address has not been initialized, returns None

        :returns: Address of head node
        :rtype: str
        """
        if not self.ray_head_address:
            self.ray_head_address = parse_ray_head_node_address(
                os.path.join(self.entities[0].path, self.entities[0].name + ".out")
            )
        return self.ray_head_address

    def get_dashboard_address(self):
        """Returns dashboard address

        The format is <head_ip>:<dashboard_port>

        :returns: Dashboard address
        :rtype: str
        """
        return self.get_head_address() + ":" + str(self.entities[0].dashboard_port)

    def _update_workers(self):
        """Update worker args before launching them."""
        for worker in range(1, len(self.entities)):
            self.entities[worker].set_head_log(
                f"{os.path.join(self.entities[0].path, self.entities[0].name)}.out"
            )


def find_ray_exe():
    """Find ray executable in current path."""
    # TODO add this to CONFIG?
    try:
        return expand_exe_path("ray")
    except (TypeError, FileNotFoundError):
        raise SmartSimError("Could not find ray executable")


def parse_ray_head_node_address(head_log):
    """Get the ray head node host address from the log file produced
    by the head process.

    :param head_log: full path to log file of head node
    :return: address of the head host
    :rtype: str
    """

    max_attempts = 24
    attempts = 0
    while not os.path.isfile(head_log):
        _time.sleep(5)
        attempts += 1
        if attempts == max_attempts:
            raise RuntimeError(f"Could not find Ray cluster head log file {head_log}")

    attempts = 0
    head_ip = None
    while head_ip is None:
        _time.sleep(5)
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
            raise RuntimeError(
                f"Could not find Ray cluster head address in log file {head_log}."
            )

    return head_ip


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
        interface="ipogif0",
        run_command=None,
        alloc=None,
        dash_port=8265,
        **kwargs,
    ):
        self.dashboard_port = dash_port
        self.batch_settings = None
        self.files = None

        run_args = init_default({}, run_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_port, ray_password, interface, ray_args
        )

        run_settings = settings.create_run_settings(
            launcher=launcher,
            exe="python",
            exe_args=ray_exe_args,
            run_args=run_args,
            run_command=run_command if run_command else "auto",
            alloc=alloc,
            **kwargs,
        )

        run_settings.set_tasks_per_node(1)
        run_settings.set_tasks(1)

        super().__init__(name, path, run_settings)

    def _build_ray_exe_args(self, ray_port, ray_password, interface, ray_args):

        # python script that launches ray head node
        ray_starter_args = [
            "-m",
            "smartsim._core.entrypoints.ray",
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


class RayWorker(SmartSimEntity):
    def __init__(
        self,
        name,
        path,
        ray_password,
        ray_port,
        run_args=None,
        ray_args=None,
        interface="ipogif0",
        launcher="slurm",
        run_command=None,
        alloc=None,
        **kwargs,
    ):

        self.batch_settings = None
        self.files = None

        run_args = init_default({}, run_args, dict)
        ray_args = init_default({}, ray_args, dict)

        ray_exe_args = self._build_ray_exe_args(
            ray_password, ray_args, ray_port, interface
        )

        run_settings = settings.create_run_settings(
            launcher=launcher,
            exe=sys.executable,
            exe_args=ray_exe_args,
            run_args=run_args,
            run_command=run_command,
            alloc=alloc,
            **kwargs,
        )

        run_settings.set_tasks_per_node(1)
        run_settings.set_tasks(1)

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
        ray_starter_args = [
            "-m",
            "smartsim._core.entrypoints.ray",
            f"+ray-exe={find_ray_exe()}",
            f"+port={ray_port}",
            f"+ifname={interface}",
        ]
        if ray_password:
            ray_starter_args += [f"+redis-password={ray_password}"]

        used = [
            "block",
            "redis-password",
            "start",
            "head",
            "port",
            "dashboard-port",
            "dashboard-host",
        ]
        extra_ray_args = []
        for key, value in ray_args.items():
            if key not in used:
                extra_ray_args += [f"+ray-args=--{key}={value}"]
        ray_starter_args += extra_ray_args

        return " ".join(ray_starter_args)
