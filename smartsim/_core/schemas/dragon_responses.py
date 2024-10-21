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

from pydantic import BaseModel, Field

import smartsim._core.schemas.utils as _utils
from smartsim.status import JobStatus

# Black and Pylint disagree about where to put the `...`
# pylint: disable=multiple-statements

response_registry = _utils.SchemaRegistry["DragonResponse"]()


class DragonResponse(BaseModel):
    error_message: t.Optional[str] = None


@response_registry.register("run")
class DragonRunResponse(DragonResponse):
    step_id: t.Annotated[str, Field(min_length=1)]


@response_registry.register("status_update")
class DragonUpdateStatusResponse(DragonResponse):
    # status is a dict: {step_id: (is_alive, returncode)}
    statuses: t.Mapping[
        t.Annotated[str, Field(min_length=1)],
        t.Tuple[JobStatus, t.Optional[t.List[int]]],
    ] = {}


@response_registry.register("stop")
class DragonStopResponse(DragonResponse): ...


@response_registry.register("handshake")
class DragonHandshakeResponse(DragonResponse):
    dragon_pid: int


@response_registry.register("bootstrap")
class DragonBootstrapResponse(DragonResponse):
    dragon_pid: int


@response_registry.register("shutdown")
class DragonShutdownResponse(DragonResponse): ...
