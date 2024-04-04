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
import typing as t
from dataclasses import dataclass, field
from threading import RLock

# pylint: disable=import-error
# isort: off
from dragon.infrastructure.policy import Policy
from dragon.native.process import Process, TemplateProcess
from dragon.native.process_group import (
    ProcessGroup,
    DragonProcessGroupError,
    Error,
    Running,
)
from dragon.native.machine import System, Node

# pylint: enable=import-error
# isort: on
from smartsim._core.schemas import (
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
from smartsim._core.utils.helpers import create_short_id_str
from smartsim.status import TERMINAL_STATUSES, SmartSimStatus

DRG_ERROR_STATUS = str(Error())
DRG_RUNNING_STATUS = str(Running())


@dataclass
class ProcessGroupInfo:
    status: SmartSimStatus
    process_group: t.Optional[ProcessGroup] = None
    puids: t.Optional[t.List[t.Optional[int]]] = None  # puids can be None
    return_codes: t.Optional[t.List[int]] = None
    hosts: t.List[str] = field(default_factory=list)

    @property
    def smartsim_info(self) -> t.Tuple[SmartSimStatus, t.Optional[t.List[int]]]:
        return (self.status, self.return_codes)


class DragonBackend:
    """The DragonBackend class is the main interface between
    SmartSim and Dragon. It is not intended to be user-facing,
    and will only be called by the Dragon entry-point script or
    by threads spawned by it.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._group_infos: t.Dict[str, ProcessGroupInfo] = {}
        self._step_id_lock = RLock()
        self._hostlist_lock = RLock()
        self._step_id = 0
        # hosts available for execution
        # dictionary maps hostname to step_id of
        # step being executed on it
        self._initialize_hosts()
        self._queued_steps: "collections.OrderedDict[str, DragonRunRequest]" = (
            collections.OrderedDict()
        )
        self._running_steps: t.List[str] = []
        self._completed_steps: t.List[str] = []

        num_hosts = len(self._hosts)
        host_string = str(num_hosts) + (" hosts" if num_hosts > 1 else " host")
        self._shutdown_requested = False
        self._updates = 0
        print(f"{host_string} available for execution: {self._hosts}")

    def print_status(self) -> None:
        print("\n-----------------------Launcher Status-----------------------")
        print(f"| {self._updates}: System hosts: ", self._hosts)
        print(f"| {self._updates}: Free hosts: ", list(self._free_hosts))
        print(f"| {self._updates}: Allocated hosts: ", self._allocated_hosts)
        print(f"| {self._updates}: Running steps: ", self._running_steps)
        print(f"| {self._updates}: Group infos: ", self._group_infos)
        print(f"| {self._updates}: There are {len(self._queued_steps)} queued steps")
        print("-------------------------------------------------------------\n")

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_requested

    def _initialize_hosts(self) -> None:
        with self._hostlist_lock:
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
        with self._hostlist_lock:
            if num_hosts <= 0 or num_hosts > len(self._free_hosts):
                return None
            to_allocate = []
            for _ in range(num_hosts):
                host = self._free_hosts.popleft()
                self._allocated_hosts[host] = step_id
                to_allocate.append(host)
            return to_allocate

    def _get_new_id(self) -> str:
        with self._step_id_lock:
            self._step_id += 1
            return create_short_id_str() + "-" + str(self._step_id)

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

        self._queued_steps[step_id] = request
        self._group_infos[step_id] = ProcessGroupInfo(
            status=SmartSimStatus.STATUS_NEVER_STARTED
        )
        return DragonRunResponse(step_id=step_id)

    def update(self) -> None:
        self._updates += 1
        started = []
        for step_id, request in self._queued_steps.items():
            hosts = self._allocate_step(step_id, self._queued_steps[step_id])
            if not hosts:
                continue

            print(f"Step id {step_id} allocated on {hosts}")

            global_policy = Policy(
                placement=Policy.Placement.HOST_NAME, host_name=hosts[0]
            )
            grp = ProcessGroup(
                restart=False, pmi_enabled=request.pmi_enabled, policy=global_policy
            )

            for node_name in hosts[: request.nodes]:
                local_policy = Policy(
                    placement=Policy.Placement.HOST_NAME, host_name=node_name
                )
                tmp_proc = TemplateProcess(
                    target=request.exe,
                    args=request.exe_args,
                    cwd=request.path,
                    env={**request.current_env, **request.env},
                    # stdout=Popen.PIPE,
                    # stderr=Popen.PIPE,
                    policy=local_policy,
                )
                grp.add_process(nproc=request.tasks_per_node, template=tmp_proc)

            grp.init()
            grp.start()
            self._group_infos[step_id] = ProcessGroupInfo(
                process_group=grp,
                puids=grp.puids,
                return_codes=[],
                status=SmartSimStatus.STATUS_RUNNING,
                hosts=hosts,
            )
            self._running_steps.append(step_id)
            started.append(step_id)

        if started:
            print(f"{self._updates}: {started=}")

        for step_id in started:
            self._queued_steps.pop(step_id)

        terminated = []
        for step_id in self._running_steps:
            group_info = self._group_infos[step_id]
            grp = group_info.process_group

            if grp.status == DRG_RUNNING_STATUS:
                group_info.status = SmartSimStatus.STATUS_RUNNING
            else:
                puids = group_info.puids
                if puids is not None and all(puid is not None for puid in puids):
                    try:
                        group_info.return_codes = [
                            Process(None, ident=puid).returncode for puid in puids
                        ]
                    except (ValueError, TypeError):
                        group_info.return_codes = [-1 for _ in puids]
                else:
                    group_info.return_codes = [0]
                group_info.status = (
                    SmartSimStatus.STATUS_FAILED
                    if any(group_info.return_codes) or grp.status == DRG_ERROR_STATUS
                    else SmartSimStatus.STATUS_COMPLETED
                )

            if group_info.status in TERMINAL_STATUSES:
                terminated.append(step_id)

        if terminated:
            print(f"{self._updates}: {terminated=}", flush=True)
        for step_id in terminated:
            self._running_steps.remove(step_id)
            self._completed_steps.append(step_id)
            group_info = self._group_infos[step_id]
            if group_info is not None:
                with self._hostlist_lock:
                    for host in group_info.hosts:
                        print(f"{self._updates}: Releasing host {host}", flush=True)
                        self._allocated_hosts.pop(host)
                        self._free_hosts.append(host)

    @process_request.register
    def _(self, request: DragonUpdateStatusRequest) -> DragonUpdateStatusResponse:
        return DragonUpdateStatusResponse(
            statuses={
                step_id: self._group_infos[step_id].smartsim_info
                for step_id in request.step_ids
                if step_id in self._group_infos
            }
        )

    @process_request.register
    def _(self, request: DragonStopRequest) -> DragonStopResponse:
        if request.step_id in self._group_infos:
            # Technically we could just terminate, but what if
            # the application intercepts that and ignores it?
            proc_group = self._group_infos[request.step_id].process_group
            if proc_group is None:
                self._group_infos[request.step_id].status = SmartSimStatus.STATUS_FAILED
            elif proc_group.status not in TERMINAL_STATUSES:
                try:
                    proc_group.kill()
                except DragonProcessGroupError:
                    try:
                        proc_group.stop()
                    except DragonProcessGroupError:
                        print("Process group already stopped")

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
        return DragonShutdownResponse()
