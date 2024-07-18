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
import time
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock

from tabulate import tabulate

# pylint: disable=import-error
# isort: off
import dragon.infrastructure.connection as dragon_connection
import dragon.infrastructure.policy as dragon_policy
import dragon.native.group_state as dragon_group_state
import dragon.native.process as dragon_process
import dragon.native.process_group as dragon_process_group
import dragon.native.machine as dragon_machine

# pylint: enable=import-error
# isort: on
from ...._core.config import get_config
from ...._core.schemas import (
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
from ...._core.utils.helpers import create_short_id_str
from ....log import get_logger
from ....status import TERMINAL_STATUSES, SmartSimStatus

logger = get_logger(__name__)


class DragonStatus(str, Enum):
    ERROR = str(dragon_group_state.Error())
    RUNNING = str(dragon_group_state.Running())

    def __str__(self) -> str:
        return self.value


@dataclass
class ProcessGroupInfo:
    status: SmartSimStatus
    """Status of step"""
    process_group: t.Optional[dragon_process_group.ProcessGroup] = None
    """Internal Process Group object, None for finished or not started steps"""
    puids: t.Optional[t.List[t.Optional[int]]] = None  # puids can be None
    """List of Process UIDS belonging to the ProcessGroup"""
    return_codes: t.Optional[t.List[int]] = None
    """List of return codes of completed processes"""
    hosts: t.List[str] = field(default_factory=list)
    """List of hosts on which the Process Group """
    redir_workers: t.Optional[dragon_process_group.ProcessGroup] = None
    """Workers used to redirect stdout and stderr to file"""

    @property
    def smartsim_info(self) -> t.Tuple[SmartSimStatus, t.Optional[t.List[int]]]:
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

    def __init__(self, pid: int) -> None:
        self._pid = pid
        """PID of dragon executable which launched this server"""
        self._group_infos: t.Dict[str, ProcessGroupInfo] = {}
        """ProcessGroup execution state information"""
        self._queue_lock = RLock()
        """Lock that needs to be acquired to access internal queues"""
        self._step_ids = (f"{create_short_id_str()}-{id}" for id in itertools.count())
        """Incremental ID to assign to new steps prior to execution"""

        self._initialize_hosts()
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
        smartsim_config = get_config()
        self._cooldown_period = (
            smartsim_config.telemetry_frequency * 2 + 5
            if smartsim_config.telemetry_enabled
            else 5
        )
        """Time in seconds needed to server to complete shutdown"""

        self._view = DragonBackendView(self)
        logger.debug(self._view.host_desc)

    @property
    def hosts(self) -> list[str]:
        with self._queue_lock:
            return self._hosts

    @property
    def allocated_hosts(self) -> dict[str, str]:
        with self._queue_lock:
            return self._allocated_hosts

    @property
    def free_hosts(self) -> t.Deque[str]:
        with self._queue_lock:
            return self._free_hosts

    @property
    def group_infos(self) -> dict[str, ProcessGroupInfo]:
        with self._queue_lock:
            return self._group_infos

    def _initialize_hosts(self) -> None:
        with self._queue_lock:
            self._nodes = [
                dragon_machine.Node(node) for node in dragon_machine.System().nodes
            ]
            self._hosts: t.List[str] = sorted(node.hostname for node in self._nodes)
            self._cpus = [node.num_cpus for node in self._nodes]
            self._gpus = [node.num_gpus for node in self._nodes]

            """List of hosts available in allocation"""
            self._free_hosts: t.Deque[str] = collections.deque(self._hosts)
            """List of hosts on which steps can be launched"""
            self._allocated_hosts: t.Dict[str, str] = {}
            """Mapping of hosts on which a step is already running to step ID"""

    def __str__(self) -> str:
        return self.status_message

    @property
    def status_message(self) -> str:
        """Message with status of available nodes and history of launched jobs.

        :returns: Status message
        """
        return (
            "Dragon server backend update\n"
            f"{self._view.host_table}\n{self._view.step_table}"
        )

    def _heartbeat(self) -> None:
        self._last_beat = self.current_time

    @property
    def cooldown_period(self) -> int:
        """Time (in seconds) the server will wait before shutting down

        when exit conditions are met (see ``should_shutdown()`` for further details).
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
        """
        if self._shutdown_requested and self._can_shutdown:
            return self._has_cooled_down
        return False

    @property
    def current_time(self) -> float:
        """Current time for DragonBackend object, in seconds since the Epoch"""
        return time.time()

    def _can_honor_policy(
        self, request: DragonRunRequest
    ) -> t.Tuple[bool, t.Optional[str]]:
        """Check if the policy can be honored with resources available
        in the allocation.
        :param request: DragonRunRequest containing policy information
        :returns: Tuple indicating if the policy can be honored and
        an optional error message"""
        # ensure the policy can be honored
        if request.policy:
            if request.policy.cpu_affinity:
                # make sure some node has enough CPUs
                available = max(self._cpus)
                requested = max(request.policy.cpu_affinity)

                if requested >= available:
                    return False, "Cannot satisfy request, not enough CPUs available"

            if request.policy.gpu_affinity:
                # make sure some node has enough GPUs
                available = max(self._gpus)
                requested = max(request.policy.gpu_affinity)

                if requested >= available:
                    return False, "Cannot satisfy request, not enough GPUs available"

        return True, None

    def _can_honor(self, request: DragonRunRequest) -> t.Tuple[bool, t.Optional[str]]:
        """Check if request can be honored with resources available in the allocation.

        Currently only checks for total number of nodes,
        in the future it will also look at other constraints
        such as memory, accelerators, and so on.
        """
        if request.nodes > len(self._hosts):
            message = f"Cannot satisfy request. Requested {request.nodes} nodes, "
            message += f"but only {len(self._hosts)} nodes are available."
            return False, message
        if self._shutdown_requested:
            message = "Cannot satisfy request, server is shutting down."
            return False, message

        honorable, err = self._can_honor_policy(request)
        if not honorable:
            return False, err

        return True, None

    def _allocate_step(
        self, step_id: str, request: DragonRunRequest
    ) -> t.Optional[t.List[str]]:

        num_hosts: int = request.nodes
        with self._queue_lock:
            if num_hosts <= 0 or num_hosts > len(self._free_hosts):
                return None
            to_allocate = []
            for _ in range(num_hosts):
                host = self._free_hosts.popleft()
                self._allocated_hosts[host] = step_id
                to_allocate.append(host)
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

                self._group_infos[step_id].status = SmartSimStatus.STATUS_CANCELLED
                self._group_infos[step_id].return_codes = [-9]

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

            affinity = dragon_policy.Policy.Affinity.DEFAULT
            cpu_affinity: t.List[int] = []
            gpu_affinity: t.List[int] = []

            # Customize policy only if the client requested it, otherwise use default
            if run_request.policy is not None:
                # Affinities are not mutually exclusive. If specified, both are used
                if run_request.policy.cpu_affinity:
                    affinity = dragon_policy.Policy.Affinity.SPECIFIC
                    cpu_affinity = run_request.policy.cpu_affinity

                if run_request.policy.gpu_affinity:
                    affinity = dragon_policy.Policy.Affinity.SPECIFIC
                    gpu_affinity = run_request.policy.gpu_affinity
            logger.debug(
                f"Affinity strategy: {affinity}, "
                f"CPU affinity mask: {cpu_affinity}, "
                f"GPU affinity mask: {gpu_affinity}"
            )
            if affinity != dragon_policy.Policy.Affinity.DEFAULT:
                return dragon_policy.Policy(
                    placement=dragon_policy.Policy.Placement.HOST_NAME,
                    host_name=node_name,
                    affinity=affinity,
                    cpu_affinity=cpu_affinity,
                    gpu_affinity=gpu_affinity,
                )

        return dragon_policy.Policy(
            placement=dragon_policy.Policy.Placement.HOST_NAME,
            host_name=node_name,
        )

    def _start_steps(self) -> None:
        self._heartbeat()
        with self._queue_lock:
            started = []
            for step_id, request in self._queued_steps.items():
                hosts = self._allocate_step(step_id, self._queued_steps[step_id])
                if not hosts:
                    continue

                logger.debug(f"Step id {step_id} allocated on {hosts}")

                global_policy = dragon_policy.Policy(
                    placement=dragon_policy.Policy.Placement.HOST_NAME,
                    host_name=hosts[0],
                )
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
                        env={**request.current_env, **request.env},
                        stdout=dragon_process.Popen.PIPE,
                        stderr=dragon_process.Popen.PIPE,
                        policy=local_policy,
                    )
                    grp.add_process(nproc=request.tasks_per_node, template=tmp_proc)

                try:
                    grp.init()
                    grp.start()
                    grp_status = SmartSimStatus.STATUS_RUNNING
                except Exception as e:
                    logger.error(e)
                    grp_status = SmartSimStatus.STATUS_FAILED

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
                    and grp_status == SmartSimStatus.STATUS_RUNNING
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
                elif puids is not None and grp_status == SmartSimStatus.STATUS_RUNNING:
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
        self._heartbeat()
        with self._queue_lock:
            terminated = []
            for step_id in self._running_steps:
                group_info = self._group_infos[step_id]
                grp = group_info.process_group
                if grp is None:
                    group_info.status = SmartSimStatus.STATUS_FAILED
                    group_info.return_codes = [-1]
                elif group_info.status not in TERMINAL_STATUSES:
                    if grp.status == str(DragonStatus.RUNNING):
                        group_info.status = SmartSimStatus.STATUS_RUNNING
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
                        if not group_info.status == SmartSimStatus.STATUS_CANCELLED:
                            group_info.status = (
                                SmartSimStatus.STATUS_FAILED
                                if any(group_info.return_codes)
                                or grp.status == DragonStatus.ERROR
                                else SmartSimStatus.STATUS_COMPLETED
                            )

                if group_info.status in TERMINAL_STATUSES:
                    terminated.append(step_id)

            if terminated:
                logger.debug(f"{terminated=}")

            for step_id in terminated:
                self._running_steps.remove(step_id)
                self._completed_steps.append(step_id)
                group_info = self._group_infos[step_id]
                if group_info is not None:
                    for host in group_info.hosts:
                        logger.debug(f"Releasing host {host}")
                        try:
                            self._allocated_hosts.pop(host)
                        except KeyError:
                            logger.error(f"Tried to free a non-allocated host: {host}")
                        self._free_hosts.append(host)
                    group_info.process_group = None
                    group_info.redir_workers = None

    def _update_shutdown_status(self) -> None:
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
        if self.current_time - self._last_update_time > 10:
            self._last_update_time = self.current_time
            return True
        return False

    def _update(self) -> None:
        self._stop_steps()
        self._start_steps()
        self._refresh_statuses()
        self._update_shutdown_status()

    def _kill_all_running_jobs(self) -> None:
        with self._queue_lock:
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
                    status=SmartSimStatus.STATUS_FAILED, return_codes=[-1]
                )
            else:
                self._queued_steps[step_id] = request
                self._group_infos[step_id] = ProcessGroupInfo(
                    status=SmartSimStatus.STATUS_NEVER_STARTED
                )
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
    def __init__(self, backend: DragonBackend):
        self._backend = backend

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

        in the allocation.
        """
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
