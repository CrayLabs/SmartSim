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

from .....log import get_logger
from ...comm.channel.channel import CommChannelBase
from ...message_handler import MessageHandler
from ...mli_schemas.response.response_capnp import ResponseBuilder

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger(__file__)


def build_failure_reply(status: "Status", message: str) -> ResponseBuilder:
    return MessageHandler.build_response(
        status=status,
        message=message,
        result=[],
        custom_attributes=None,
    )


def exception_handler(
    exc: Exception, reply_channel: t.Optional[CommChannelBase], failure_message: str
) -> None:
    """
    Logs exceptions and sends a failure response.

    :param exc: The exception to be logged
    :param reply_channel: The channel used to send replies
    :param failure_message: Failure message to log and send back
    """
    logger.exception(
        f"{failure_message}\n"
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {str(exc)}"
    )
    if reply_channel:
        serialized_resp = MessageHandler.serialize_response(
            build_failure_reply("fail", failure_message)
        )
        reply_channel.send(serialized_resp)
    else:
        logger.warning("Unable to notify client of error without a reply channel")
