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

import zmq
import zmq.auth.thread

from smartsim._core.config.config import get_config
from smartsim._core.schemas import dragon_requests as _dragonRequests
from smartsim._core.schemas import dragon_responses as _dragonResponses
from smartsim._core.schemas import utils as _utils
from smartsim._core.utils.security import KeyManager
from smartsim.log import get_logger

if t.TYPE_CHECKING:
    from zmq import Context
    from zmq.sugar.socket import Socket

logger = get_logger(__name__)

AUTHENTICATOR: t.Optional["zmq.auth.thread.ThreadAuthenticator"] = None


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
    context: "zmq.Context[t.Any]",
    socket_type: int,
    is_server: bool,
) -> "Socket[t.Any]":
    """Create secured socket that consumes & produces encrypted messages

    :param context: ZMQ context object
    :param socket_type: Type of ZMQ socket to create
    :param is_server: Pass `True` to secure the socket as server. Pass `False`
    to secure the socket as a client.
    :returns: the secured socket prepared for sending encrypted messages
    """
    config = get_config()
    socket: "Socket[t.Any]" = context.socket(socket_type)

    key_manager = KeyManager(config, as_server=is_server, as_client=not is_server)
    server_keys, client_keys = key_manager.get_keys()
    logger.debug(f"Applying keys to socket: {server_keys}, {client_keys}")

    if is_server:
        logger.debug("Configuring socket as server")

        # configure the server keys on the socket
        socket.curve_secretkey = server_keys.private
        socket.curve_publickey = server_keys.public

        socket.curve_server = True
    else:
        # configure client keys on the socket to encrypt outgoing messages
        socket.curve_secretkey = client_keys.private
        socket.curve_publickey = client_keys.public

        # set the server public key for decrypting incoming messages
        socket.curve_serverkey = server_keys.public
    return socket


def get_authenticator(
    context: "zmq.Context[t.Any]", timeout: int = get_config().dragon_server_timeout
) -> "zmq.auth.thread.ThreadAuthenticator":
    """Create an authenticator to handle encryption of ZMQ communications

    :param context: ZMQ context object
    :returns: the activated `Authenticator`
    """
    # pylint: disable-next=global-statement
    global AUTHENTICATOR

    if AUTHENTICATOR is not None:
        if AUTHENTICATOR.is_alive():
            return AUTHENTICATOR
        try:
            logger.debug("Stopping authenticator")
            AUTHENTICATOR.thread.authenticator.zap_socket.close()
            AUTHENTICATOR.thread.join(0.1)
            AUTHENTICATOR = None
        except Exception as e:
            logger.debug(e)
        finally:
            logger.debug("Stopped authenticator")

    config = get_config()

    key_manager = KeyManager(config, as_client=True)
    server_keys, client_keys = key_manager.get_keys()
    logger.debug(f"Applying keys to authenticator: {server_keys}, {client_keys}")

    AUTHENTICATOR = zmq.auth.thread.ThreadAuthenticator(context, log=logger)

    ctx_sndtimeo = context.getsockopt(zmq.SNDTIMEO)
    ctx_rcvtimeo = context.getsockopt(zmq.RCVTIMEO)

    AUTHENTICATOR.context.setsockopt(zmq.SNDTIMEO, timeout)
    AUTHENTICATOR.context.setsockopt(zmq.RCVTIMEO, timeout)
    AUTHENTICATOR.context.setsockopt(zmq.REQ_CORRELATE, 1)
    AUTHENTICATOR.context.setsockopt(zmq.REQ_RELAXED, 1)

    # allow all keys in the client key directory to connect
    logger.debug(f"Securing with client keys in {key_manager.client_keys_dir}")
    AUTHENTICATOR.configure_curve(domain="*", location=key_manager.client_keys_dir)

    logger.debug("Starting authenticator")
    AUTHENTICATOR.start()

    context.setsockopt(zmq.SNDTIMEO, ctx_sndtimeo)
    context.setsockopt(zmq.RCVTIMEO, ctx_rcvtimeo)

    return AUTHENTICATOR
