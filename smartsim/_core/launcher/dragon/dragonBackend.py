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
import time
import typing as t
from dataclasses import dataclass, field
from threading import RLock

# pylint: disable=import-error
# isort: off
from dragon.infrastructure.connection import Connection
from dragon.infrastructure.policy import Policy
from dragon.native.process import Process, ProcessTemplate, Popen
from dragon.native.process_group import (
    ProcessGroup,
    DragonProcessGroupError,
)
from dragon.native.machine import System, Node

# pylint: enable=import-error
# isort: on
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

DRG_ERROR_STATUS = "Error"
DRG_RUNNING_STATUS = "Running"

logger = get_logger(__name__)


@dataclass
class ProcessGroupInfo:
    status: SmartSimStatus
    process_group: t.Optional[ProcessGroup] = None
    puids: t.Optional[t.List[t.Optional[int]]] = None  # puids can be None
    return_codes: t.Optional[t.List[int]] = None
    hosts: t.List[str] = field(default_factory=list)
    redir_workers: t.Optional[ProcessGroup] = None

    @property
    def smartsim_info(self) -> t.Tuple[SmartSimStatus, t.Optional[t.List[int]]]:
        return (self.status, self.return_codes)


# Thanks to Colin Wahl from HPE HPC Dragon Team
def redir_worker(io_conn: Connection, file_path: str) -> None:
    """Read stdout/stderr from the Dragon connection.

    :param io_conn: Dragon connection to stdout or stderr
    :type io_conn: Connection
    :param file_path: path to file to write to
    :type file_path: str
    """
    try:
        with open(file_path, "a", encoding="utf-8") as file_to_write:
            while True:
                output = io_conn.recv()
                print(output, flush=True, file=file_to_write, end="")
    except EOFError:
        pass
    finally:
        io_conn.close()


class DragonBackend:
    """The DragonBackend class is the main interface between
    SmartSim and Dragon. It is not intended to be user-facing,
    and will only be called by the Dragon entry-point script or
    by threads spawned by it.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._group_infos: t.Dict[str, ProcessGroupInfo] = {}
        self._queue_lock = RLock()
        self._step_id = 0
        # hosts available for execution
        # dictionary maps hostname to step_id of
        # step being executed on it
        self._initialize_hosts()
        self._queued_steps: "collections.OrderedDict[str, DragonRunRequest]" = (
            collections.OrderedDict()
        )
        self._stop_requests: t.Deque[DragonStopRequest] = collections.deque()
        self._running_steps: t.List[str] = []
        self._completed_steps: t.List[str] = []
        self._last_beat: float = 0.0
        self._heartbeat()
        self._last_update_time = self._last_beat
        num_hosts = len(self._hosts)
        host_string = str(num_hosts) + (" hosts" if num_hosts > 1 else " host")
        self._shutdown_requested = False
        self._can_shutdown = False
        self._frontend_shutdown: bool = False
        logger.debug(f"{host_string} available for execution: {self._hosts}")

    def __str__(self) -> str:
        return self.get_status_message()

    def get_status_message(self) -> str:
        msg = [f"System hosts: {self._hosts}"]
        msg.append(f"Free hosts: {list(self._free_hosts)}")
        msg.append(f"Allocated hosts: {self._allocated_hosts}")
        msg.append(f"Running steps: {self._running_steps}")
        msg.append(f"Group infos: {self._group_infos}")
        msg.append(f"There are {len(self._queued_steps)} queued steps")
        return "\n".join(msg)

    def _heartbeat(self) -> None:
        self._last_beat = self.current_time

    @property
    def frontend_shutdown(self) -> bool:
        return self._frontend_shutdown

    @property
    def last_heartbeat(self) -> float:
        return self._last_beat

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_requested and self._can_shutdown

    @property
    def current_time(self) -> float:
        return time.time_ns() / 1e9

    def _initialize_hosts(self) -> None:
        with self._queue_lock:
            self._hosts: t.List[str] = sorted(
                Node(node).hostname for node in System().nodes
            )
            self._free_hosts: t.Deque[str] = collections.deque(self._hosts)
            self._allocated_hosts: t.Dict[str, str] = {}

    def _can_honor(self, request: DragonRunRequest) -> t.Tuple[bool, t.Optional[str]]:
        """Check if request can be honored with resources
        available in the allocation.
        Currently only checks for total number of nodes,
        in the future it will also look at other constraints
        such as memory, accelerators, and so on.
        """
        if request.nodes > len(self._hosts):
            message = f"Cannot satisfy request. Requested {request.nodes} nodes, "
            message += f"but only {len(self._hosts)} nodes are available."
            return False, message
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

    def _get_new_id(self) -> str:
        step_id = create_short_id_str() + "-" + str(self._step_id)
        self._step_id += 1
        return step_id

    @functools.singledispatchmethod
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use
    def process_request(self, request: DragonRequest) -> DragonResponse:
        raise TypeError(f"Unsure how to process a `{type(request)}` request")

    @process_request.register
    def _(self, request: DragonRunRequest) -> DragonRunResponse:

        step_id = self._get_new_id()
        honorable, err = self._can_honor(request)
        if not honorable:
            self._group_infos[step_id] = ProcessGroupInfo(
                status=SmartSimStatus.STATUS_FAILED, return_codes=[-1]
            )
            return DragonRunResponse(step_id=step_id, error_message=err)

        with self._queue_lock:
            self._queued_steps[step_id] = request
            self._group_infos[step_id] = ProcessGroupInfo(
                status=SmartSimStatus.STATUS_NEVER_STARTED
            )
            return DragonRunResponse(step_id=step_id)

    @staticmethod
    def _start_redirect_workers(
        global_policy: Policy,
        policies: t.List[Policy],
        puids: t.List[int],
        out_file: t.Optional[str],
        err_file: t.Optional[str],
    ) -> ProcessGroup:
        grp_redir = ProcessGroup(restart=False, policy=global_policy)
        for pol, puid in zip(policies, puids):
            proc = Process(None, ident=puid)
            if out_file:
                grp_redir.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=redir_worker,
                        args=(proc.stdout_conn, out_file),
                        stdout=Popen.DEVNULL,
                        policy=pol,
                    ),
                )
            if err_file:
                grp_redir.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=redir_worker,
                        args=(proc.stderr_conn, err_file),
                        stdout=Popen.DEVNULL,
                        policy=pol,
                    ),
                )
        try:
            grp_redir.init()
            time.sleep(0.1)
            grp_redir.start()
        except Exception as e:
            raise IOError(
                f"Could not redirect stdout and stderr for PUIDS {puids}"
            ) from e

        return grp_redir

    def _stop_steps(self) -> None:
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
                        and proc_group.status == DRG_RUNNING_STATUS
                    ):
                        try:
                            proc_group.kill()
                        except DragonProcessGroupError:
                            try:
                                proc_group.stop()
                            except DragonProcessGroupError:
                                logger.error("Process group already stopped")
                    redir_group = self._group_infos[step_id].redir_workers
                    if redir_group is not None:
                        try:
                            redir_group.join(0.1)
                            del redir_group
                        except Exception as e:
                            logger.error(e)

                self._group_infos[step_id].status = SmartSimStatus.STATUS_CANCELLED
                self._group_infos[step_id].return_codes = [-9]

    def _start_steps(self) -> None:
        started = []
        with self._queue_lock:
            for step_id, request in self._queued_steps.items():
                hosts = self._allocate_step(step_id, self._queued_steps[step_id])
                if not hosts:
                    continue

                logger.debug(f"Step id {step_id} allocated on {hosts}")

                global_policy = Policy(
                    placement=Policy.Placement.HOST_NAME, host_name=hosts[0]
                )
                grp = ProcessGroup(
                    restart=False, pmi_enabled=request.pmi_enabled, policy=global_policy
                )

                policies = []
                for node_name in hosts:
                    local_policy = Policy(
                        placement=Policy.Placement.HOST_NAME, host_name=node_name
                    )
                    policies.extend([local_policy] * request.tasks_per_node)
                    tmp_proc = ProcessTemplate(
                        target=request.exe,
                        args=request.exe_args,
                        cwd=request.path,
                        env={**request.current_env, **request.env},
                        stdout=Popen.PIPE,
                        stderr=Popen.PIPE,
                        policy=local_policy,
                    )
                    grp.add_process(nproc=request.tasks_per_node, template=tmp_proc)

                try:
                    grp.init()
                    grp.start()
                except Exception as e:
                    logger.error(e)

                puids = None
                try:
                    puids = grp.puids
                    self._group_infos[step_id] = ProcessGroupInfo(
                        process_group=grp,
                        puids=puids,
                        return_codes=[],
                        status=SmartSimStatus.STATUS_RUNNING,
                        hosts=hosts,
                    )
                    self._running_steps.append(step_id)
                    started.append(step_id)
                except Exception as e:
                    logger.error(e)

                if puids is not None:
                    try:
                        redir_grp = DragonBackend._start_redirect_workers(
                            global_policy,
                            policies,
                            puids,
                            request.output_file,
                            request.error_file,
                        )
                        self._group_infos[step_id].redir_workers = redir_grp
                    except Exception as e:
                        logger.error(e)

            if started:
                logger.debug(f"{started=}")

            for step_id in started:
                try:
                    self._queued_steps.pop(step_id)
                except KeyError:
                    logger.error(
                        "Tried to allocate the same step twice, step id {step_id}"
                    )

    def _refresh_statuses(self) -> None:
        terminated = []
        with self._queue_lock:
            for step_id in self._running_steps:
                group_info = self._group_infos[step_id]
                grp = group_info.process_group
                if grp is None:
                    group_info.status = SmartSimStatus.STATUS_FAILED
                    group_info.return_codes = [-1]
                elif group_info.status not in TERMINAL_STATUSES:
                    if grp.status == DRG_RUNNING_STATUS:
                        group_info.status = SmartSimStatus.STATUS_RUNNING
                    else:
                        puids = group_info.puids
                        if puids is not None and all(
                            puid is not None for puid in puids
                        ):
                            try:
                                group_info.return_codes = [
                                    Process(None, ident=puid).returncode
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
                                or grp.status == DRG_ERROR_STATUS
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
                            logger.error(f"Tried to free same host twice: {host}")
                        self._free_hosts.append(host)
                    group_info.process_group = None
                    group_info.redir_workers = None

    def _update_shutdown_status(self) -> None:
        self._can_shutdown = all(
            grp_info.status in TERMINAL_STATUSES
            and grp_info.process_group is None
            and grp_info.redir_workers is None
            for grp_info in self._group_infos.values()
        )

    def _should_print_status(self) -> bool:
        if self._last_beat - self._last_update_time > 10:
            self._last_update_time = self._last_beat
            return True
        return False

    def update(self) -> None:
        logger.debug("Dragon Backend update thread started")
        while not self.should_shutdown:
            try:
                self._heartbeat()
                self._stop_steps()
                self._start_steps()
                self._refresh_statuses()
                self._update_shutdown_status()
            except Exception as e:
                logger.error(e)
            if self._should_print_status():
                try:
                    logger.debug(str(self))
                except ValueError as e:
                    logger.error(e)
        logger.debug("Dragon Backend update thread stopping")

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
        self._can_shutdown |= request.immediate
        self._frontend_shutdown = request.frontend_shutdown
        return DragonShutdownResponse()
