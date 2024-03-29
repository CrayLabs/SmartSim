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

import zmq.auth
import zmq.auth.thread

from smartsim._core.config.config import get_config
from smartsim._core.schemas import dragonRequests as _dragonRequests
from smartsim._core.schemas import dragonResponses as _dragonResponses
from smartsim._core.schemas import utils as _utils
from smartsim._core.utils.security import KeyManager

if t.TYPE_CHECKING:
    from zmq import Context, SocketType
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


def get_secure_socket(
    context: "Context[t.Any]",
    socket_type: "SocketType",
    is_server: bool,
    authenticator: t.Optional[zmq.auth.Authenticator] = None,
) -> "t.Tuple[Socket[t.Any], zmq.auth.Authenticator]":
    """Create secured socket that consumes & produces encrypted messages

    :param context: ZMQ context object
    :type context: zmq.Context
    :param socket_type: Type of ZMQ socket to create
    :type socket_type: zmq.SocketType
    :param is_server: Pass `True` to secure the socket as server. Pass `False`
    to secure the socket as a client.
    :type is_server: bool
    :param authenticator: (optional) An existing authenticator that will be used
    to authenticate secure communications.
    :type authenticator: Optional[zmq.auth.Authenticator]
    :returns: the secured socket prepared for sending encrypted messages and
    an active authenticator if one was not supplied as a parameter.
    :rtype: Tuple[zmq.Socket, zmq.auth.Authenticator]"""
    config = get_config()
    socket = context.socket(socket_type)

    key_manager = KeyManager(config, as_server=is_server, as_client=not is_server)
    server_keys, client_keys = key_manager.get_keys()

    # start an auth thread to provide encryption services on the socket
    if authenticator is None:
        authenticator = zmq.auth.thread.ThreadAuthenticator(context)

    if is_server:
        # allow all keys in the client key directory to connect
        authenticator.configure_curve(
            domain="*"
        )  # , location=key_manager.client_key_dir

        # configure the server keys on the socket
        socket.curve_secretkey = server_keys.private
        socket.curve_publickey = client_keys.public
        socket.curve_server = True
    else:
        # configure client keys on the socket to encrypt outgoing messages
        socket.curve_secretkey = client_keys.private
        socket.curve_publickey = client_keys.public

        # set the server public key for decrypting incoming messages
        socket.curve_serverkey = client_keys.public

    if authenticator is None:
        authenticator.start()

    return socket, authenticator
