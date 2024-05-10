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

import typing as t

from pydantic import BaseModel, Field, PositiveInt

import smartsim._core.schemas.utils as _utils

# Black and Pylint disagree about where to put the `...`
# pylint: disable=multiple-statements

request_registry = _utils.SchemaRegistry["DragonRequest"]()


class DragonRequest(BaseModel): ...


class DragonRunRequestView(DragonRequest):
    exe: t.Annotated[str, Field(min_length=1)]
    exe_args: t.List[t.Annotated[str, Field(min_length=1)]] = []
    path: t.Annotated[str, Field(min_length=1)]
    nodes: PositiveInt = 1
    tasks: PositiveInt = 1
    tasks_per_node: PositiveInt = 1
    hostlist: t.Optional[t.Annotated[str, Field(min_length=1)]] = None
    output_file: t.Optional[t.Annotated[str, Field(min_length=1)]] = None
    error_file: t.Optional[t.Annotated[str, Field(min_length=1)]] = None
    env: t.Dict[str, t.Optional[str]] = {}
    name: t.Optional[t.Annotated[str, Field(min_length=1)]] = None
    pmi_enabled: bool = True


@request_registry.register("run")
class DragonRunRequest(DragonRunRequestView):
    current_env: t.Dict[str, t.Optional[str]] = {}

    def __str__(self) -> str:
        return str(DragonRunRequestView.parse_obj(self.dict(exclude={"current_env"})))


@request_registry.register("update_status")
class DragonUpdateStatusRequest(DragonRequest):
    step_ids: t.List[t.Annotated[str, Field(min_length=1)]]


@request_registry.register("stop")
class DragonStopRequest(DragonRequest):
    step_id: t.Annotated[str, Field(min_length=1)]


@request_registry.register("handshake")
class DragonHandshakeRequest(DragonRequest): ...


@request_registry.register("bootstrap")
class DragonBootstrapRequest(DragonRequest):
    address: t.Annotated[str, Field(min_length=1)]


@request_registry.register("shutdown")
class DragonShutdownRequest(DragonRequest):
    immediate: bool = True
    """Whether the server should shut down immediately, setting this to False means
    that the server will shut down when all jobs are terminated."""
    frontend_shutdown: bool = True
    """Whether the frontend will have to shut down or wait for external termination"""
