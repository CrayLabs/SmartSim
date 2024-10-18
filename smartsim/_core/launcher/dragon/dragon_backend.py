# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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
import collections
import functools
import itertools
import os
import socket
import time
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock

from tabulate import tabulate

# pylint: disable=import-error,C0302,R0915
# isort: off

import dragon.infrastructure.connection as dragon_connection
import dragon.infrastructure.policy as dragon_policy
import dragon.infrastructure.process_desc as dragon_process_desc

import dragon.native.process as dragon_process
import dragon.native.process_group as dragon_process_group
import dragon.native.machine as dragon_machine

from smartsim._core.launcher.dragon.pqueue import NodePrioritizer, PrioritizerFilter
from smartsim._core.mli.infrastructure.control.listener import (
    ConsumerRegistrationListener,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_util import create_ddict
from smartsim.error.errors import SmartSimError

# pylint: enable=import-error
# isort: on
from ....log import get_logger
from ....status import TERMINAL_STATUSES, JobStatus
from ...config import get_config
from ...schemas import (
    DragonHandshakeRequest,
    DragonHandshakeResponse,
    DragonRequest,
    DragonResponse,
    DragonRunRequest,
    DragonRunResponse,
    DragonShutdownRequest,
    DragonShutdownResponse,
    DragonStopRequest,
    DragonStopResponse,
    DragonUpdateStatusRequest,
    DragonUpdateStatusResponse,
)
from ...utils.helpers import create_short_id_str

logger = get_logger(__name__)


class DragonStatus(str, Enum):
    ERROR = "Error"
    RUNNING = "Running"

    def __str__(self) -> str:
        return self.value


@dataclass
class ProcessGroupInfo:
    status: JobStatus
    """Status of step"""
    process_group: t.Optional[dragon_process_group.ProcessGroup] = None
    """Internal Process Group object, None for finished or not started steps"""
    puids: t.Optional[t.List[t.Optional[int]]] = None  # puids can be None
    """List of Process UIDS belonging to the ProcessGroup"""
    return_codes: t.Optional[t.List[int]] = None
    """List of return codes of completed processes"""
    hosts: t.List[str] = field(default_factory=list)
    """List of hosts on which the Process Group should be executed"""
    redir_workers: t.Optional[dragon_process_group.ProcessGroup] = None
    """Workers used to redirect stdout and stderr to file"""

    @property
    def smartsim_info(self) -> t.Tuple[JobStatus, t.Optional[t.List[int]]]:
        """Information needed by SmartSim Launcher and Job Manager"""
        return (self.status, self.return_codes)

    def __str__(self) -> str:
        if self.process_group is not None and self.redir_workers is not None:
            msg = [f"Active Group ({self.status})"]
            if self.puids is not None:
                msg.append(f"Number processes: {len(self.puids)}")
        else:
            msg = [f"Inactive Group ({self.status})"]

        if self.hosts is not None:
            msg.append(f"Hosts: {','.join(self.hosts)}")
        if self.return_codes is not None:
            msg.append(f"{self.return_codes}")

        return ", ".join(msg)


# Thanks to Colin Wahl from HPE HPC Dragon Team
def redir_worker(io_conn: dragon_connection.Connection, file_path: str) -> None:
    """Read stdout/stderr from the Dragon connection.

    :param io_conn: Dragon connection to stdout or stderr
    :param file_path: path to file to write to
    """
    while io_conn is None or not io_conn.readable:
        time.sleep(0.1)
    try:
        with open(file_path, "a", encoding="utf-8") as file_to_write:
            while True:
                output = io_conn.recv()
                print(output, flush=True, file=file_to_write, end="")
    except EOFError:
        pass
    except Exception as e:
        print(e)
    finally:
        try:
            io_conn.close()
        except Exception as e:
            print(e)


class DragonBackend:
    """The DragonBackend class is the main interface between
    SmartSim and Dragon. It is not intended to be user-facing,
    and will only be called by the Dragon entry-point script or
    by threads spawned by it.
    """

    _DEFAULT_NUM_MGR_PER_NODE = 2
    """The default number of manager processes for each feature store node"""
    _DEFAULT_MEM_PER_NODE = 512 * 1024**2
    """The default memory capacity (in bytes) to allocate for a feaure store node"""

    def __init__(self, pid: int) -> None:
        self._pid = pid
        """PID of dragon executable which launched this server"""
        self._group_infos: t.Dict[str, ProcessGroupInfo] = {}
        """ProcessGroup execution state information"""
        self._queue_lock = RLock()
        """Lock that needs to be acquired to access internal queues"""
        self._step_ids = (f"{create_short_id_str()}-{id}" for id in itertools.count())
        """Incremental ID to assign to new steps prior to execution"""

        self._queued_steps: "collections.OrderedDict[str, DragonRunRequest]" = (
            collections.OrderedDict()
        )
        """Steps waiting for execution"""
        self._stop_requests: t.Deque[DragonStopRequest] = collections.deque()
        """Stop requests which have not been processed yet"""
        self._running_steps: t.List[str] = []
        """List of currently running steps"""
        self._completed_steps: t.List[str] = []
        """List of completed steps"""
        self._last_beat: float = 0.0
        """Time at which the last heartbeat was set"""
        self._heartbeat()
        self._last_update_time = self._last_beat
        """Time at which the status update was printed the last time"""
        self._shutdown_requested = False
        """Whether the shutdown was requested to this server"""
        self._can_shutdown = False
        """Whether the server can shut down"""
        self._frontend_shutdown: bool = False
        """Whether the server frontend should shut down when the backend does"""
        self._shutdown_initiation_time: t.Optional[float] = None
        """The time at which the server initiated shutdown"""
        self._cooldown_period = self._initialize_cooldown()
        """Time in seconds needed by the server to complete shutdown"""
        self._backbone: t.Optional[BackboneFeatureStore] = None
        """The backbone feature store"""
        self._listener: t.Optional[dragon_process.Process] = None
        """The standalone process executing the event consumer"""

        self._nodes: t.List["dragon_machine.Node"] = []
        """Node capability information for hosts in the allocation"""
        self._hosts: t.List[str] = []
        """List of hosts available in allocation"""
        self._cpus: t.List[int] = []
        """List of cpu-count by node"""
        self._gpus: t.List[int] = []
        """List of gpu-count by node"""
        self._allocated_hosts: t.Dict[str, t.Set[str]] = {}
        """Mapping with hostnames as keys and a set of running step IDs as the value"""

        self._initialize_hosts()
        self._prioritizer = NodePrioritizer(self._nodes, self._queue_lock)

    @property
    def hosts(self) -> list[str]:
        with self._queue_lock:
            return self._hosts

    @property
    def allocated_hosts(self) -> dict[str, t.Set[str]]:
        """A map of host names to the step id executing on a host

        :returns: Dictionary with host name as key and step id as value"""
        with self._queue_lock:
            return self._allocated_hosts

    @property
    def free_hosts(self) -> t.Sequence[str]:
        """Find hosts that do not have a step assigned

        :returns: List of host names"""
        with self._queue_lock:
            return list(map(lambda x: x.hostname, self._prioritizer.unassigned()))

    @property
    def group_infos(self) -> dict[str, ProcessGroupInfo]:
        """Find information pertaining to process groups executing on a host

        :returns: Dictionary with host name as key and group information as value"""
        with self._queue_lock:
            return self._group_infos

    def _initialize_hosts(self) -> None:
        """Prepare metadata about the allocation"""
        with self._queue_lock:
            self._nodes = [
                dragon_machine.Node(node) for node in dragon_machine.System().nodes
            ]
            self._hosts = sorted(node.hostname for node in self._nodes)
            self._cpus = [node.num_cpus for node in self._nodes]
            self._gpus = [node.num_gpus for node in self._nodes]
            self._allocated_hosts = collections.defaultdict(set)

    def __str__(self) -> str:
        return self.status_message

    @property
    def status_message(self) -> str:
        """Message with status of available nodes and history of launched jobs.

        :returns: a status message
        """
        view = DragonBackendView(self)
        return "Dragon server backend update\n" f"{view.host_table}\n{view.step_table}"

    def _heartbeat(self) -> None:
        """Update the value of the last heartbeat to the current time."""
        self._last_beat = self.current_time

    @property
    def cooldown_period(self) -> int:
        """Time (in seconds) the server will wait before shutting down when
        exit conditions are met (see ``should_shutdown()`` for further details).
        """
        return self._cooldown_period

    @property
    def _has_cooled_down(self) -> bool:
        if self._shutdown_initiation_time is None:
            logger.debug(f"Starting cooldown period of {self._cooldown_period} seconds")
            self._shutdown_initiation_time = self.current_time
        return (
            self.current_time - self._shutdown_initiation_time > self._cooldown_period
        )

    @property
    def frontend_shutdown(self) -> bool:
        """Whether the frontend will have to shutdown once the backend does

        If False, the frontend will wait for an external signal to stop.
        """
        return self._frontend_shutdown

    @property
    def last_heartbeat(self) -> float:
        """Time (in seconds) at which the last heartbeat was set"""
        return self._last_beat

    @property
    def should_shutdown(self) -> bool:
        """Whether the server should shut down

        A server should shut down if a DragonShutdownRequest was received
        and it requested immediate shutdown, or if it did not request immediate
        shutdown, but all jobs have been executed.
        In both cases, a cooldown period may need to be waited before shutdown.

        :returns: `True` if the server should terminate, otherwise `False`
        """
        if self._shutdown_requested and self._can_shutdown:
            return self._has_cooled_down
        return False

    @property
    def current_time(self) -> float:
        """Current time for DragonBackend object, in seconds since the Epoch

        :returns: the current timestamp"""
        return time.time()

    def _can_honor_policy(
        self, request: DragonRunRequest
    ) -> t.Tuple[bool, t.Optional[str]]:
        """Check if the policy can be honored with resources available
        in the allocation.

        :param request: `DragonRunRequest` to validate
        :returns: Tuple indicating if the policy can be honored and
        an optional error message"""
        # ensure the policy can be honored
        if request.policy:
            logger.debug(f"{request.policy=}{self._cpus=}{self._gpus=}")

            if request.policy.cpu_affinity:
                # make sure some node has enough CPUs
                last_available = max(self._cpus or [-1])
                requested = max(request.policy.cpu_affinity)
                if not any(self._cpus) or requested >= last_available:
                    return False, "Cannot satisfy request, not enough CPUs available"
            if request.policy.gpu_affinity:
                # make sure some node has enough GPUs
                last_available = max(self._gpus or [-1])
                requested = max(request.policy.gpu_affinity)
                if not any(self._gpus) or requested >= last_available:
                    logger.warning(
                        f"failed check w/{self._gpus=}, {requested=}, {last_available=}"
                    )
                    return False, "Cannot satisfy request, not enough GPUs available"
        return True, None

    def _can_honor(self, request: DragonRunRequest) -> t.Tuple[bool, t.Optional[str]]:
        """Check if request can be honored with resources available in
        the allocation. Currently only checks for total number of nodes,
        in the future it will also look at other constraints such as memory,
        accelerators, and so on.

        :param request: `DragonRunRequest` to validate
        :returns: Tuple indicating if the request can be honored and
        an optional error message
        """
        honorable, err = self._can_honor_state(request)
        if not honorable:
            return False, err

        honorable, err = self._can_honor_policy(request)
        if not honorable:
            return False, err

        honorable, err = self._can_honor_hosts(request)
        if not honorable:
            return False, err

        return True, None

    def _can_honor_hosts(
        self, request: DragonRunRequest
    ) -> t.Tuple[bool, t.Optional[str]]:
        """Check if the current state of the backend process inhibits executing
        the request.

        :param request: `DragonRunRequest` to validate
        :returns: Tuple indicating if the request can be honored and
        an optional error message"""
        all_hosts = frozenset(self._hosts)
        num_nodes = request.nodes

        # fail if requesting more nodes than the total number available
        if num_nodes > len(all_hosts):
            message = f"Cannot satisfy request. {num_nodes} requested nodes"
            message += f" exceeds {len(all_hosts)} available."
            return False, message

        requested_hosts = all_hosts
        if request.hostlist:
            requested_hosts = frozenset(
                {host.strip() for host in request.hostlist.split(",")}
            )

        valid_hosts = all_hosts.intersection(requested_hosts)
        invalid_hosts = requested_hosts - valid_hosts

        logger.debug(f"{num_nodes=}{valid_hosts=}{invalid_hosts=}")

        if invalid_hosts:
            logger.warning(f"Some invalid hostnames were requested: {invalid_hosts}")

        # fail if requesting specific hostnames and there aren't enough available
        if num_nodes > len(valid_hosts):
            message = f"Cannot satisfy request. Requested {num_nodes} nodes, "
            message += f"but only {len(valid_hosts)} named hosts are available."
            return False, message

        return True, None

    def _can_honor_state(
        self, _request: DragonRunRequest
    ) -> t.Tuple[bool, t.Optional[str]]:
        """Check if the current state of the backend process inhibits executing
        the request.
        :param _request: the DragonRunRequest to verify
        :returns: Tuple indicating if the request can be honored and
        an optional error message"""
        if self._shutdown_requested:
            message = "Cannot satisfy request, server is shutting down."
            return False, message

        return True, None

    def _allocate_step(
        self, step_id: str, request: DragonRunRequest
    ) -> t.Optional[t.List[str]]:
        """Identify the hosts on which the request will be executed

        :param step_id: The identifier of a step that will be executed on the host
        :param request: The request to be executed
        :returns: A list of selected hostnames"""
        # ensure at least one host is selected
        num_hosts: int = request.nodes

        with self._queue_lock:
            if num_hosts <= 0 or num_hosts > len(self._hosts):
                logger.debug(
                    f"The number of requested hosts ({num_hosts}) is invalid or"
                    f" cannot be satisfied with {len(self._hosts)} available nodes"
                )
                return None

            hosts = []
            if request.hostlist:
                # convert the comma-separated argument into a real list
                hosts = [host for host in request.hostlist.split(",") if host]

            filter_on: t.Optional[PrioritizerFilter] = None
            if request.policy and request.policy.gpu_affinity:
                filter_on = PrioritizerFilter.GPU

            nodes = self._prioritizer.next_n(num_hosts, filter_on, step_id, hosts)

            if len(nodes) < num_hosts:
                # exit if the prioritizer can't identify enough nodes
                return None

            to_allocate = [node.hostname for node in nodes]

            for hostname in to_allocate:
                # track assigning this step to each node
                self._allocated_hosts[hostname].add(step_id)

            return to_allocate

    @staticmethod
    def _create_redirect_workers(
        global_policy: dragon_policy.Policy,
        policies: t.List[dragon_policy.Policy],
        puids: t.List[int],
        out_file: t.Optional[str],
        err_file: t.Optional[str],
    ) -> dragon_process_group.ProcessGroup:
        grp_redir = dragon_process_group.ProcessGroup(
            restart=False, policy=global_policy, pmi_enabled=False
        )
        for pol, puid in zip(policies, puids):
            proc = dragon_process.Process(None, ident=puid)
            if out_file:
                grp_redir.add_process(
                    nproc=1,
                    template=dragon_process.ProcessTemplate(
                        target=redir_worker,
                        args=(proc.stdout_conn, out_file),
                        stdout=dragon_process.Popen.DEVNULL,
                        policy=pol,
                    ),
                )
            if err_file:
                grp_redir.add_process(
                    nproc=1,
                    template=dragon_process.ProcessTemplate(
                        target=redir_worker,
                        args=(proc.stderr_conn, err_file),
                        stdout=dragon_process.Popen.DEVNULL,
                        policy=pol,
                    ),
                )

        return grp_redir

    def _stop_steps(self) -> None:
        """Trigger termination of all currently executing steps"""
        self._heartbeat()
        with self._queue_lock:
            while len(self._stop_requests) > 0:
                request = self._stop_requests.popleft()
                step_id = request.step_id
                if step_id not in self._group_infos:
                    logger.error(f"Requested to stop non-existing step {step_id}")
                    continue

                logger.debug(f"Stopping step {step_id}")
                if request.step_id in self._queued_steps:
                    self._queued_steps.pop(step_id)
                else:
                    # Technically we could just terminate, but what if
                    # the application intercepts that and ignores it?
                    proc_group = self._group_infos[step_id].process_group
                    if (
                        proc_group is not None
                        and proc_group.status == DragonStatus.RUNNING
                    ):
                        try:
                            proc_group.kill()
                        except dragon_process_group.DragonProcessGroupError:
                            try:
                                proc_group.stop()
                            except dragon_process_group.DragonProcessGroupError:
                                logger.error("Process group already stopped")
                    redir_group = self._group_infos[step_id].redir_workers
                    if redir_group is not None:
                        try:
                            redir_group.join(0.1)
                            redir_group = None
                        except Exception as e:
                            logger.error(e)

                self._group_infos[step_id].status = JobStatus.CANCELLED
                self._group_infos[step_id].return_codes = [-9]

    def _create_backbone(self) -> BackboneFeatureStore:
        """
        Creates a BackboneFeatureStore if one does not exist. Updates
        environment variables of this process to include the backbone
        descriptor.

        :returns: The backbone feature store
        """
        if self._backbone is None:
            backbone_storage = create_ddict(
                len(self._hosts),
                self._DEFAULT_NUM_MGR_PER_NODE,
                self._DEFAULT_MEM_PER_NODE,
            )

            self._backbone = BackboneFeatureStore(
                backbone_storage, allow_reserved_writes=True
            )

            # put the backbone descriptor in the env vars
            os.environ.update(self._backbone.get_env())

        return self._backbone

    @staticmethod
    def _initialize_cooldown() -> int:
        """Load environment configuration and determine the correct cooldown
        period to apply to the backend process.

        :returns: The calculated cooldown (in seconds)
        """
        smartsim_config = get_config()
        return (
            smartsim_config.telemetry_frequency * 2 + 5
            if smartsim_config.telemetry_enabled
            else 5
        )

    def start_event_listener(
        self, cpu_affinity: list[int], gpu_affinity: list[int]
    ) -> dragon_process.Process:
        """Start a standalone event listener.

        :param cpu_affinity: The CPU affinity for the process
        :param gpu_affinity: The GPU affinity for the process
        :returns: The dragon Process managing the process
        :raises SmartSimError: If the backbone is not provided
        """
        if self._backbone is None:
            raise SmartSimError("Backbone feature store is not available")

        service = ConsumerRegistrationListener(
            self._backbone, 1.0, 2.0, as_service=True, health_check_frequency=90
        )

        options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
        local_policy = dragon_policy.Policy(
            placement=dragon_policy.Policy.Placement.HOST_NAME,
            host_name=socket.gethostname(),
            cpu_affinity=cpu_affinity,
            gpu_affinity=gpu_affinity,
        )
        process = dragon_process.Process(
            target=service.execute,
            args=[],
            cwd=os.getcwd(),
            env={
                **os.environ,
                **self._backbone.get_env(),
            },
            policy=local_policy,
            options=options,
            stderr=dragon_process.Popen.STDOUT,
            stdout=dragon_process.Popen.STDOUT,
        )
        process.start()
        return process

    @staticmethod
    def create_run_policy(
        request: DragonRequest, node_name: str
    ) -> "dragon_policy.Policy":
        """Create a dragon Policy from the request and node name

        :param request: DragonRunRequest containing policy information
        :param node_name: Name of the node on which the process will run
        :returns: dragon_policy.Policy object mapped from request properties"""
        if isinstance(request, DragonRunRequest):
            run_request: DragonRunRequest = request

            cpu_affinity: t.List[int] = []
            gpu_affinity: t.List[int] = []

            # Customize policy only if the client requested it, otherwise use default
            if run_request.policy is not None:
                # Affinities are not mutually exclusive. If specified, both are used
                if run_request.policy.cpu_affinity:
                    cpu_affinity = run_request.policy.cpu_affinity

                if run_request.policy.gpu_affinity:
                    gpu_affinity = run_request.policy.gpu_affinity
            logger.debug(
                f"CPU affinity mask: {cpu_affinity}, "
                f"GPU affinity mask: {gpu_affinity}"
            )
            return dragon_policy.Policy(
                placement=dragon_policy.Policy.Placement.HOST_NAME,
                host_name=node_name,
                cpu_affinity=cpu_affinity,
                gpu_affinity=gpu_affinity,
            )

        return dragon_policy.Policy(
            placement=dragon_policy.Policy.Placement.HOST_NAME,
            host_name=node_name,
        )

    def _start_steps(self) -> None:
        """Start all new steps created since the last update."""
        self._heartbeat()

        with self._queue_lock:
            started = []
            for step_id, request in self._queued_steps.items():
                hosts = self._allocate_step(step_id, self._queued_steps[step_id])
                if not hosts:
                    continue

                logger.debug(f"Step id {step_id} allocated on {hosts}")

                global_policy = self.create_run_policy(request, hosts[0])
                options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
                grp = dragon_process_group.ProcessGroup(
                    restart=False, pmi_enabled=request.pmi_enabled, policy=global_policy
                )

                policies = []
                for node_name in hosts:
                    local_policy = self.create_run_policy(request, node_name)
                    policies.extend([local_policy] * request.tasks_per_node)
                    tmp_proc = dragon_process.ProcessTemplate(
                        target=request.exe,
                        args=request.exe_args,
                        cwd=request.path,
                        env={
                            **request.current_env,
                            **request.env,
                            **(self._backbone.get_env() if self._backbone else {}),
                        },
                        stdout=dragon_process.Popen.PIPE,
                        stderr=dragon_process.Popen.PIPE,
                        policy=local_policy,
                        options=options,
                    )
                    grp.add_process(nproc=request.tasks_per_node, template=tmp_proc)

                try:
                    grp.init()
                    grp.start()
                    grp_status = JobStatus.RUNNING
                except Exception as e:
                    logger.error(e)
                    grp_status = JobStatus.FAILED

                puids = None
                try:
                    puids = list(
                        set(grp.puids + [puid for puid, retcode in grp.inactive_puids])
                    )
                    self._group_infos[step_id] = ProcessGroupInfo(
                        process_group=grp,
                        puids=puids,
                        return_codes=[],
                        status=grp_status,
                        hosts=hosts,
                    )
                    self._running_steps.append(step_id)
                    started.append(step_id)
                except Exception as e:
                    logger.error(e)

                if (
                    puids is not None
                    and len(puids) == len(policies)
                    and grp_status == JobStatus.RUNNING
                ):
                    redir_grp = DragonBackend._create_redirect_workers(
                        global_policy,
                        policies,
                        puids,
                        request.output_file,
                        request.error_file,
                    )
                    try:
                        redir_grp.init()
                        redir_grp.start()
                    except Exception as e:
                        raise IOError(
                            f"Could not redirect stdout and stderr for PUIDS {puids}"
                        ) from e
                    self._group_infos[step_id].redir_workers = redir_grp
                elif puids is not None and grp_status == JobStatus.RUNNING:
                    logger.error("Cannot redirect workers: some PUIDS are missing")

            if started:
                logger.debug(f"{started=}")

            for step_id in started:
                try:
                    self._queued_steps.pop(step_id)
                except KeyError:
                    logger.error(
                        f"Tried to allocate the same step twice, step id {step_id}"
                    )
                except Exception as e:
                    logger.error(e)

    def _refresh_statuses(self) -> None:
        """Query underlying management system for step status and update
        stored assigned and unassigned task information"""
        self._heartbeat()
        with self._queue_lock:
            terminated: t.Set[str] = set()
            for step_id in self._running_steps:
                group_info = self._group_infos[step_id]
                grp = group_info.process_group
                if grp is None:
                    group_info.status = JobStatus.FAILED
                    group_info.return_codes = [-1]
                elif group_info.status not in TERMINAL_STATUSES:
                    if grp.status == str(DragonStatus.RUNNING):
                        group_info.status = JobStatus.RUNNING
                    else:
                        puids = group_info.puids
                        if puids is not None and all(
                            puid is not None for puid in puids
                        ):
                            try:
                                group_info.return_codes = [
                                    dragon_process.Process(None, ident=puid).returncode
                                    for puid in puids
                                ]
                            except (ValueError, TypeError) as e:
                                logger.error(e)
                                group_info.return_codes = [-1 for _ in puids]
                        else:
                            group_info.return_codes = [0]
                        if not group_info.status == JobStatus.CANCELLED:
                            group_info.status = (
                                JobStatus.FAILED
                                if any(group_info.return_codes)
                                or grp.status == DragonStatus.ERROR
                                else JobStatus.COMPLETED
                            )

                if group_info.status in TERMINAL_STATUSES:
                    terminated.add(step_id)

            if terminated:
                logger.debug(f"{terminated=}")

            # remove all the terminated steps from all hosts
            for host in list(self._allocated_hosts.keys()):
                self._allocated_hosts[host].difference_update(terminated)

            for step_id in terminated:
                self._running_steps.remove(step_id)
                self._completed_steps.append(step_id)
                group_info = self._group_infos[step_id]
                if group_info is not None:
                    for host in group_info.hosts:
                        logger.debug(f"Releasing host {host}")
                        if host not in self._allocated_hosts:
                            logger.error(f"Tried to free a non-allocated host: {host}")
                        else:
                            # remove any hosts that have had all their steps terminated
                            if not self._allocated_hosts[host]:
                                self._allocated_hosts.pop(host)
                        self._prioritizer.decrement(host, step_id)
                    group_info.process_group = None
                    group_info.redir_workers = None

    def _update_shutdown_status(self) -> None:
        """Query the status of running tasks and update the status
        of any that have completed.
        """
        self._heartbeat()
        with self._queue_lock:
            self._can_shutdown |= (
                all(
                    grp_info.status in TERMINAL_STATUSES
                    and grp_info.process_group is None
                    and grp_info.redir_workers is None
                    for grp_info in self._group_infos.values()
                )
                and self._shutdown_requested
            )

    def _should_print_status(self) -> bool:
        """Determine if status messages should be printed based off the last
        update. Returns `True` to trigger prints, `False` otherwise.
        """
        if self.current_time - self._last_update_time > 10:
            self._last_update_time = self.current_time
            return True
        return False

    def _update(self) -> None:
        """Trigger all update queries and update local state database"""
        self._create_backbone()

        self._stop_steps()
        self._start_steps()
        self._refresh_statuses()
        self._update_shutdown_status()

    def _kill_all_running_jobs(self) -> None:
        with self._queue_lock:
            if self._listener and self._listener.is_alive:
                self._listener.kill()

            for step_id, group_info in self._group_infos.items():
                if group_info.status not in TERMINAL_STATUSES:
                    self._stop_requests.append(DragonStopRequest(step_id=step_id))

    def update(self) -> None:
        """Update internal data structures, queues, and job statuses"""
        logger.debug("Dragon Backend update thread started")
        while not self.should_shutdown:
            try:
                self._update()
                time.sleep(0.1)
            except Exception as e:
                logger.error(e)
            if self._should_print_status():
                try:
                    logger.debug(str(self))
                except ValueError as e:
                    logger.error(e)

        logger.debug("Dragon Backend update thread stopping")

    @functools.singledispatchmethod
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use
    def process_request(self, request: DragonRequest) -> DragonResponse:
        """Process an incoming DragonRequest"""
        raise TypeError(f"Unsure how to process a `{type(request)}` request")

    @process_request.register
    def _(self, request: DragonRunRequest) -> DragonRunResponse:
        step_id = next(self._step_ids)
        with self._queue_lock:
            honorable, err = self._can_honor(request)
            if not honorable:
                self._group_infos[step_id] = ProcessGroupInfo(
                    status=JobStatus.FAILED, return_codes=[-1]
                )
            else:
                self._queued_steps[step_id] = request
                self._group_infos[step_id] = ProcessGroupInfo(status=JobStatus.NEW)
            return DragonRunResponse(step_id=step_id, error_message=err)

    @process_request.register
    def _(self, request: DragonUpdateStatusRequest) -> DragonUpdateStatusResponse:
        with self._queue_lock:
            return DragonUpdateStatusResponse(
                statuses={
                    step_id: self._group_infos[step_id].smartsim_info
                    for step_id in request.step_ids
                    if step_id in self._group_infos
                }
            )

    @process_request.register
    def _(self, request: DragonStopRequest) -> DragonStopResponse:
        with self._queue_lock:
            self._stop_requests.append(request)
        return DragonStopResponse()

    @process_request.register
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use,unused-argument
    def _(self, request: DragonHandshakeRequest) -> DragonHandshakeResponse:
        return DragonHandshakeResponse(dragon_pid=self._pid)

    @process_request.register
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use,unused-argument
    def _(self, request: DragonShutdownRequest) -> DragonShutdownResponse:
        self._shutdown_requested = True
        self._update_shutdown_status()
        if request.immediate:
            self._kill_all_running_jobs()
        self._frontend_shutdown = request.frontend_shutdown
        return DragonShutdownResponse()


class DragonBackendView:
    def __init__(self, backend: DragonBackend) -> None:
        """Initialize the instance

        :param backend: A dragon backend used to produce the view"""
        self._backend = backend
        """A dragon backend used to produce the view"""

        logger.debug(self.host_desc)

    @property
    def host_desc(self) -> str:
        hosts = self._backend.hosts
        num_hosts = len(hosts)
        host_string = str(num_hosts) + (" hosts" if num_hosts != 1 else " host")
        return f"{host_string} available for execution: {hosts}"

    @staticmethod
    def _proc_group_info_table_line(
        step_id: str, proc_group_info: ProcessGroupInfo
    ) -> t.List[str]:
        table_line = [step_id, f"{proc_group_info.status.value}"]

        if proc_group_info.hosts is not None:
            table_line.append(f"{','.join(proc_group_info.hosts)}")
        else:
            table_line.append("")

        if proc_group_info.return_codes is not None:
            table_line.append(
                f"{','.join(str(ret) for ret in proc_group_info.return_codes)}"
            )
        else:
            table_line.append("")

        if proc_group_info.puids is not None:
            table_line.append(f"{len(proc_group_info.puids)}")
        else:
            table_line.append("")

        return table_line

    @property
    def step_table(self) -> str:
        """Table representation of all jobs which have been started on the server."""
        headers = ["Step", "Status", "Hosts", "Return codes", "Num procs"]

        group_infos = self._backend.group_infos

        colalign = (
            ["left", "left", "left", "center", "center"]
            if len(group_infos) > 0
            else None
        )
        values = [
            self._proc_group_info_table_line(step, group_info)
            for step, group_info in group_infos.items()
        ]

        return tabulate(
            values,
            headers,
            disable_numparse=True,
            tablefmt="github",
            colalign=colalign,
        )

    @property
    def host_table(self) -> str:
        """Table representation of current state of nodes available
        in the allocation."""
        headers = ["Host", "Status"]
        hosts = self._backend.hosts
        free_hosts = self._backend.free_hosts

        def _host_table_line(host: str) -> list[str]:
            return [host, "Free" if host in free_hosts else "Busy"]

        colalign = ["left", "center"] if len(hosts) > 0 else None
        values = [_host_table_line(host) for host in hosts]

        return tabulate(
            values, headers, disable_numparse=True, tablefmt="github", colalign=colalign
        )
