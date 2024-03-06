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

from pydantic import BaseModel

import smartsim._core.schemas.utils as _utils
from smartsim._core.schemas.types import NonEmptyStr

# Black and Pylint disagree about where to put the `...`
# pylint: disable=multiple-statements


class DragonResponse(BaseModel):
    error_message: t.Optional[str] = None


response_serializer = _utils.SchemaSerializer[str, DragonResponse]("response_type")


@response_serializer.register("run")
class DragonRunResponse(DragonResponse):
    step_id: NonEmptyStr


@response_serializer.register("status_update")
class DragonUpdateStatusResponse(DragonResponse):
    # status is a dict: {step_id: (is_alive, returncode)}
    statuses: t.Mapping[NonEmptyStr, t.Tuple[NonEmptyStr, t.Optional[t.List[int]]]] = {}


@response_serializer.register("stop")
class DragonStopResponse(DragonResponse): ...


@response_serializer.register("handshake")
class DragonHandshakeResponse(DragonResponse): ...


@response_serializer.register("bootstrap")
class DragonBootstrapResponse(DragonResponse): ...


@response_serializer.register("shutdown")
class DragonShutdownResponse(DragonResponse): ...
