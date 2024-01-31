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

# mypy: disable-error-code="valid-type"

import typing as t

from pydantic import BaseModel, constr


class DragonResponse(BaseModel):
    response_type: constr(min_length=1)
    error_message: t.Optional[str] = None


class DragonRunResponse(DragonResponse):
    response_type: constr(min_length=1) = "run"
    step_id: constr(min_length=1)


class DragonUpdateStatusResponse(DragonResponse):
    response_type: constr(min_length=1) = "status_update"
    # status is a dict: {step_id: (is_alive, returncode)}
    statuses: t.Mapping[
        constr(min_length=1), t.Tuple[constr(min_length=1), t.Optional[t.List[int]]]
    ] = {}


class DragonStopResponse(DragonResponse):
    response_type: constr(min_length=1) = "stop"


class DragonHandshakeResponse(DragonResponse):
    response_type: constr(min_length=1) = "handshake"


class DragonBootstrapResponse(DragonResponse):
    response_type: constr(min_length=1) = "bootstrap"
