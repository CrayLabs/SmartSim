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
        self._request_to_function: t.Mapping[
            str, t.Callable[[t.Any], DragonResponse]
        ] = {
            "run": self.run,
            "update_status": self.update_status,
            "stop": self.stop,
            "handshake": DragonBackend.handshake,
        }
        self._proc_groups: t.Dict[str, t.Tuple[ProcessGroup, t.List[int]]] = {}
        self._step_id_lock = RLock()
        self._step_id = 0

    def _get_new_id(self) -> str:
        with self._step_id_lock:
            self._step_id += 1
            return str(self._step_id)

    def process_request(self, request: DragonRequest) -> DragonResponse:
        req_type = DragonRequest.parse_obj(request).request_type
        if not req_type:
            raise ValueError("Malformed request contains empty ``request_type`` field.")
        if req_type not in self._request_to_function:
            raise ValueError(f"Unknown request type {req_type}.")

        return self._request_to_function[req_type](request)

    def run(self, request: DragonRunRequest) -> DragonRunResponse:
        run_request = DragonRunRequest.parse_obj(request)

        proc = TemplateProcess(
            target=run_request.exe,
            args=run_request.exe_args,
            cwd=run_request.path,
            env={**run_request.current_env, **run_request.env},
            # stdout=Popen.PIPE,
            # stderr=Popen.PIPE,
        )

        grp = ProcessGroup(restart=False, pmi_enabled=True)
        grp.add_process(nproc=run_request.tasks, template=proc)
        step_id = self._get_new_id()
        grp.init()
        grp.start()
        self._proc_groups[step_id] = (grp, grp.puids)

        return DragonRunResponse(step_id=step_id)

    def update_status(
        self, request: DragonUpdateStatusRequest
    ) -> DragonUpdateStatusResponse:
        update_status_request = DragonUpdateStatusRequest.parse_obj(request)

        updated_statuses: t.Dict[str, t.Tuple[str, t.Optional[t.List[int]]]] = {}
        for step_id in update_status_request.step_ids:
            return_codes: t.List[int] = []
            if step_id in self._proc_groups:
                proc_group_tuple = self._proc_groups[step_id]
                if proc_group_tuple[0].status == "Running":
                    updated_statuses[step_id] = (STATUS_RUNNING, return_codes)
                else:
                    if all(proc_id is not None for proc_id in proc_group_tuple[1]):
                        return_codes = [
                            Process(None, ident=puid).returncode
                            for puid in proc_group_tuple[1]
                        ]
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

    def stop(self, request: DragonStopRequest) -> DragonStopResponse:
        stop_request = DragonStopRequest.parse_obj(request)

        if stop_request.step_id in self._proc_groups:
            # Technically we could just terminate, but what if
            # the application intercepts that and ignores it?
            proc_group = self._proc_groups[stop_request.step_id][0]
            if proc_group.status == "Running":
                proc_group.kill()

        return DragonStopResponse()

    @staticmethod
    def handshake(request: DragonHandshakeRequest) -> DragonHandshakeResponse:
        DragonHandshakeRequest.parse_obj(request)

        return DragonHandshakeResponse()
