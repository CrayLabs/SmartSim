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

from smartsim._core.schemas import dragonRequests as _dragonRequests
from smartsim._core.schemas import dragonResponses as _dragonResponses
from smartsim._core.schemas import utils as _utils

if t.TYPE_CHECKING:
    from zmq.sugar.socket import Socket


def as_server(
    socket: "Socket[t.Any]",
) -> _utils.SocketSchemaTranslator[
    _dragonResponses.DragonResponse,
    _dragonRequests.DragonRequest,
]:
    return _utils.SocketSchemaTranslator(
        socket, _dragonResponses.response_registry, _dragonRequests.request_registry
    )


def as_client(
    socket: "Socket[t.Any]",
) -> _utils.SocketSchemaTranslator[
    _dragonRequests.DragonRequest,
    _dragonResponses.DragonResponse,
]:
    return _utils.SocketSchemaTranslator(
        socket, _dragonRequests.request_registry, _dragonResponses.response_registry
    )
