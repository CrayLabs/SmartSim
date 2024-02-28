# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import functools
import typing as t
from threading import RLock

# pylint: disable=import-error
from dragon.native.process import Process, TemplateProcess
from dragon.native.process_group import ProcessGroup

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
from smartsim.status import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEVER_STARTED,
    STATUS_RUNNING,
)

# pylint: enable=import-error


class DragonBackend:
    """The DragonBackend class is the main interface between
    SmartSim and Dragon. It is not intended to be user-facing,
    and will only be called by the Dragon entry-point script or
    by threads spawned by it.
    """

    def __init__(self) -> None:
        self._proc_groups: t.Dict[str, t.Tuple[ProcessGroup, t.List[int]]] = {}
        self._step_id_lock = RLock()
        self._step_id = 0

    def _get_new_id(self) -> str:
        with self._step_id_lock:
            self._step_id += 1
            return str(self._step_id)

    @functools.singledispatchmethod
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use
    def process_request(self, request: DragonRequest) -> DragonResponse:
        raise TypeError("Unsure how to process a `{type(request)}` request")

    @process_request.register
    def _(self, request: DragonRunRequest) -> DragonRunResponse:
        proc = TemplateProcess(
            target=request.exe,
            args=request.exe_args,
            cwd=request.path,
            env={**request.current_env, **request.env},
            # stdout=Popen.PIPE,
            # stderr=Popen.PIPE,
        )

        grp = ProcessGroup(restart=False, pmi_enabled=request.pmi_enabled)
        grp.add_process(nproc=request.tasks, template=proc)
        step_id = self._get_new_id()
        grp.init()
        grp.start()
        self._proc_groups[step_id] = (grp, grp.puids)

        return DragonRunResponse(step_id=step_id)

    @process_request.register
    def _(self, request: DragonUpdateStatusRequest) -> DragonUpdateStatusResponse:
        updated_statuses: t.Dict[str, t.Tuple[str, t.Optional[t.List[int]]]] = {}
        for step_id in request.step_ids:
            return_codes: t.List[int] = []
            if step_id in self._proc_groups:
                proc_group_tuple = self._proc_groups[step_id]
                if proc_group_tuple[0].status == "Running":
                    updated_statuses[step_id] = (STATUS_RUNNING, return_codes)
                else:
                    if all(proc_id is not None for proc_id in proc_group_tuple[1]):
                        try:
                            return_codes = [
                                Process(None, ident=puid).returncode
                                for puid in proc_group_tuple[1]
                            ]
                        except (ValueError, TypeError):
                            return_codes = [-1 for _ in proc_group_tuple[1]]
                    else:
                        return_codes = [0]
                    status = (
                        STATUS_FAILED
                        if any(return_codes) or proc_group_tuple[0].status == "Error"
                        else STATUS_COMPLETED
                    )
                    updated_statuses[step_id] = (status, return_codes)
            else:
                updated_statuses[step_id] = (STATUS_NEVER_STARTED, return_codes)

        return DragonUpdateStatusResponse(statuses=updated_statuses)

    @process_request.register
    def _(self, request: DragonStopRequest) -> DragonStopResponse:
        if request.step_id in self._proc_groups:
            # Technically we could just terminate, but what if
            # the application intercepts that and ignores it?
            proc_group = self._proc_groups[request.step_id][0]
            if proc_group.status == "Running":
                proc_group.kill()

        return DragonStopResponse()

    @process_request.register
    # Deliberately suppressing errors so that overloads have the same signature
    # pylint: disable-next=no-self-use,unused-argument
    def _(self, request: DragonHandshakeRequest) -> DragonHandshakeResponse:
        return DragonHandshakeResponse()

    @staticmethod
    def shutdown(request: DragonShutdownRequest) -> DragonShutdownResponse:
        DragonShutdownRequest.parse_obj(request)

        return DragonShutdownResponse()
