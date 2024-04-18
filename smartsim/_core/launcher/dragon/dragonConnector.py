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

from __future__ import annotations

import atexit
import fileinput
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import typing as t
from pathlib import Path
from threading import RLock

import psutil
import zmq
import zmq.auth.thread

from smartsim._core.launcher.dragon import dragonSockets
from smartsim.error.errors import SmartSimError

from ....log import get_logger
from ...config import CONFIG
from ...schemas import (
    DragonBootstrapRequest,
    DragonBootstrapResponse,
    DragonHandshakeRequest,
    DragonHandshakeResponse,
    DragonRequest,
    DragonResponse,
    DragonShutdownRequest,
)
from ...utils.network import find_free_port, get_best_interface_and_address

logger = get_logger(__name__)

_SchemaT = t.TypeVar("_SchemaT", bound=t.Union[DragonRequest, DragonResponse])

DRG_LOCK = RLock()


class DragonConnector:
    """This class encapsulates the functionality needed
    to start a Dragon server and communicate with it.
    """

    def __init__(self, graceful_cleanup: bool = True) -> None:
        super().__init__()
        self._context: zmq.Context[t.Any] = zmq.Context.instance()
        self._context.setsockopt(zmq.REQ_CORRELATE, 1)
        self._context.setsockopt(zmq.REQ_RELAXED, 1)
        self._authenticator: t.Optional[zmq.auth.thread.ThreadAuthenticator] = None
        self._timeout = CONFIG.dragon_server_timeout
        self._startup_timeout = CONFIG.dragon_server_startup_timeout
        self._set_timeout(self._timeout)
        self._dragon_head_socket: t.Optional[zmq.Socket[t.Any]] = None
        self._dragon_head_process: t.Optional[subprocess.Popen[bytes]] = None
        # Returned by dragon head, useful if shutdown is to be requested
        # but process was started by another connector
        self._dragon_head_pid: t.Optional[int] = None
        self._dragon_server_path = CONFIG.dragon_server_path
        self._graceful_cleanup = graceful_cleanup
        logger.debug(f"Dragon Server path was set to {self._dragon_server_path}")
        if self._dragon_server_path is None:
            raise SmartSimError(
                "DragonConnector could not find the dragon server path. "
                "This should not happen if the Connector was started by an "
                "experiment.\nIf the DragonConnector was started manually, "
                "then the environment variable SMARTSIM_DRAGON_SERVER_PATH "
                "should be set to an existing directory."
            )

    @property
    def is_connected(self) -> bool:
        return self._dragon_head_socket is not None

    def _handshake(self, address: str) -> None:
        self._dragon_head_socket = dragonSockets.get_secure_socket(
            self._context, zmq.REQ, False
        )
        self._dragon_head_socket.connect(address)
        try:
            dragon_handshake = _assert_schema_type(
                self.send_request(DragonHandshakeRequest()), DragonHandshakeResponse
            )
            self._dragon_head_pid = dragon_handshake.dragon_pid
            logger.debug(
                f"Successful handshake with Dragon server at address {address}"
            )
        except (zmq.ZMQError, zmq.Again) as e:
            logger.debug(e)
            self._dragon_head_socket.close()
            self._dragon_head_socket = None

            raise SmartSimError(
                f"Unsuccessful handshake with Dragon server at address {address}"
            ) from e

    def _set_timeout(self, timeout: int) -> None:
        self._context.setsockopt(zmq.SNDTIMEO, value=timeout)
        self._context.setsockopt(zmq.RCVTIMEO, value=timeout)
        if self._authenticator is not None and self._authenticator.thread is not None:
            try:
                self._authenticator.thread.authenticator.zap_socket.setsockopt(
                    zmq.SNDTIMEO, timeout
                )
                self._authenticator.thread.authenticator.zap_socket.setsockopt(
                    zmq.RCVTIMEO, timeout
                )
            except zmq.ZMQError:
                pass

    def ensure_connected(self) -> None:
        if not self.is_connected:
            self.connect_to_dragon()
        if not self.is_connected:
            raise SmartSimError("Could not connect to Dragon server")

    def _get_new_authenticator(self) -> None:
        if self._authenticator is not None:
            if self._authenticator.thread is not None:
                try:
                    logger.debug("Closing ZAP socket")
                    self._authenticator.thread.authenticator.zap_socket.close()
                except Exception as e:
                    logger.debug(f"Could not close ZAP socket, {e}")
            try:
                self._authenticator.stop()
            except zmq.Again:
                logger.debug("Could not stop authenticator")
        try:
            self._authenticator = dragonSockets.get_authenticator(self._context)
            return
        except RuntimeError as e:
            logger.error("Could not get authenticator")
            raise e from None

    # pylint: disable-next=too-many-statements,too-many-locals
    def connect_to_dragon(self) -> None:
        with DRG_LOCK:
            # TODO use manager instead
            if self.is_connected:
                return
            if self._dragon_server_path is None:
                raise SmartSimError("Path to Dragon server not set.")

            path = _resolve_dragon_path(self._dragon_server_path)
            dragon_config_log = path / CONFIG.dragon_log_filename

            if dragon_config_log.is_file():

                dragon_confs = self._parse_launched_dragon_server_info_from_files(
                    [dragon_config_log]
                )
                logger.debug(dragon_confs)
                for dragon_conf in dragon_confs:
                    if not "address" in dragon_conf:
                        continue
                    logger.debug(
                        "Found dragon server config file. Checking if the server"
                        f" is still up at address {dragon_conf['address']}."
                    )
                    try:
                        self._set_timeout(self._timeout)
                        self._get_new_authenticator()
                        self._handshake(dragon_conf["address"])
                    except SmartSimError as e:
                        logger.debug(e)
                    finally:
                        self._set_timeout(self._timeout)
                    if self.is_connected:
                        logger.debug("Connected to existing Dragon server")
                        return

            path.mkdir(parents=True, exist_ok=True)

            cmd = [
                "dragon",
                "-t",
                CONFIG.dragon_transport,
                "-l",
                CONFIG.dragon_log_level,
                sys.executable,
                "-m",
                "smartsim._core.entrypoints.dragon",
            ]

            address = get_best_interface_and_address().address
            socket_addr = ""
            connector_socket: t.Optional[zmq.Socket[t.Any]] = None
            if address is not None:
                self._set_timeout(self._startup_timeout)
                self._get_new_authenticator()
                connector_socket = dragonSockets.get_secure_socket(
                    self._context, zmq.REP, True
                )

                # find first available port >= 5995
                port = find_free_port(start=5995)
                socket_addr = f"tcp://{address}:{port}"
                logger.debug(f"Binding connector to {socket_addr}")

                connector_socket.bind(socket_addr)
                cmd += ["+launching_address", socket_addr]

            dragon_out_file = path / "dragon_head.out"
            dragon_err_file = path / "dragon_head.err"

            with open(dragon_out_file, "w", encoding="utf-8") as dragon_out, open(
                dragon_err_file, "w", encoding="utf-8"
            ) as dragon_err:
                current_env = os.environ.copy()
                current_env.update({"PYTHONUNBUFFERED": "1"})
                logger.debug(f"Starting Dragon environment: {' '.join(cmd)}")

                # pylint: disable-next=consider-using-with
                self._dragon_head_process = subprocess.Popen(
                    args=cmd,
                    bufsize=0,
                    stderr=dragon_err.fileno(),
                    stdout=dragon_out.fileno(),
                    cwd=path,
                    shell=False,
                    env=current_env,
                    start_new_session=True,
                )

            if connector_socket is None:
                raise SmartSimError("Socket failed to initialize")

            def log_dragon_outputs() -> None:
                if self._dragon_head_process:
                    self._dragon_head_process.wait(1.0)
                    if self._dragon_head_process.stdout:
                        for line in iter(
                            self._dragon_head_process.stdout.readline, b""
                        ):
                            logger.info(line.decode("utf-8").rstrip())
                    if self._dragon_head_process.stderr:
                        for line in iter(
                            self._dragon_head_process.stderr.readline, b""
                        ):
                            logger.warning(line.decode("utf-8").rstrip())
                    logger.warning(self._dragon_head_process.returncode)

            if address is not None:
                server = dragonSockets.as_server(connector_socket)
                logger.debug(f"Listening to {socket_addr}")
                request = _assert_schema_type(server.recv(), DragonBootstrapRequest)

                logger.debug(f"Connecting to {request.address}")
                server.send(
                    DragonBootstrapResponse(dragon_pid=self._dragon_head_process.pid)
                )

                connector_socket.close()
                self._set_timeout(self._timeout)
                self._handshake(request.address)

                # Only the Connector which started the server is
                # responsible of it, that's why we register the
                # cleanup in this code branch.
                # The cleanup function should not have references
                # to this object to avoid Garbage Collector lockup
                server_socket = self._dragon_head_socket
                server_process_pid = self._dragon_head_process.pid

                if server_socket is not None and self._dragon_head_process is not None:
                    atexit.register(
                        _dragon_cleanup,
                        server_socket=server_socket,
                        server_process_pid=server_process_pid,
                        server_authenticator=self._authenticator,
                        graceful=self._graceful_cleanup,
                    )
            else:
                # TODO parse output file
                log_dragon_outputs()
                raise SmartSimError("Could not receive address of Dragon head process")

    def cleanup(self, graceful: bool = True) -> None:
        if self._dragon_head_socket is not None and self._dragon_head_pid is not None:
            _dragon_cleanup(
                server_socket=self._dragon_head_socket,
                server_process_pid=self._dragon_head_pid,
                server_authenticator=self._authenticator,
                graceful=graceful,
            )
            self._dragon_head_socket = None
            self._dragon_head_pid = 0
            self._authenticator = None

    def send_request(self, request: DragonRequest, flags: int = 0) -> DragonResponse:
        self.ensure_connected()
        if (socket := self._dragon_head_socket) is None:
            raise SmartSimError("Not connected to Dragon")
        return self._send_req_with_socket(socket, request, flags)

    @staticmethod
    def _parse_launched_dragon_server_info_from_iterable(
        stream: t.Iterable[str], num_dragon_envs: t.Optional[int] = None
    ) -> t.List[t.Dict[str, str]]:
        lines = (line.strip() for line in stream)
        lines = (line for line in lines if line)
        tokenized = (line.split(maxsplit=1) for line in lines)
        tokenized = (tokens for tokens in tokenized if len(tokens) > 1)
        dragon_env_jsons = (
            config_dict
            for first, config_dict in tokenized
            if "DRAGON_SERVER_CONFIG" in first
        )
        dragon_envs = [json.loads(config_dict) for config_dict in dragon_env_jsons]

        if num_dragon_envs:
            sliced_dragon_envs = itertools.islice(dragon_envs, num_dragon_envs)
            return list(sliced_dragon_envs)
        return dragon_envs

    @classmethod
    def _parse_launched_dragon_server_info_from_files(
        cls,
        file_paths: t.List[t.Union[str, "os.PathLike[str]"]],
        num_dragon_envs: t.Optional[int] = None,
    ) -> t.List[t.Dict[str, str]]:
        with fileinput.FileInput(file_paths) as ifstream:
            dragon_envs = cls._parse_launched_dragon_server_info_from_iterable(
                ifstream, num_dragon_envs
            )

            return dragon_envs

    @staticmethod
    def _send_req_with_socket(
        socket: zmq.Socket[t.Any], request: DragonRequest, flags: int = 0
    ) -> DragonResponse:
        client = dragonSockets.as_client(socket)
        with DRG_LOCK:
            logger.debug(f"Sending {type(request).__name__}: {request}")
            client.send(request, flags)
            response = client.recv()

            logger.debug(f"Received {type(response).__name__}: {response}")
            return response


def _assert_schema_type(obj: object, typ: t.Type[_SchemaT], /) -> _SchemaT:
    if not isinstance(obj, typ):
        raise TypeError(f"Expected schema of type `{typ}`, but got {type(obj)}")
    return obj


def _dragon_cleanup(
    server_socket: t.Optional[zmq.Socket[t.Any]] = None,
    server_process_pid: t.Optional[int] = 0,
    server_authenticator: t.Optional[zmq.auth.thread.ThreadAuthenticator] = None,
    graceful: bool = True,
) -> None:
    """Clean up resources used by the launcher.
    :param server_socket: (optional) Socket used to connect to dragon environment
    :type server_socket: Optional[zmq.Socket]
    :param server_process_pid: (optional) Process ID of the dragon entrypoint
    :type server_process_pid: Optional[int]
    :param server_authenticator: (optional) Authenticator used to secure sockets
    :type server_authenticator: Optional[zmq.auth.thread.ThreadAuthenticator]
    """
    try:
        if server_socket is not None:
            print("Sending shutdown request to dragon environment")
            # pylint: disable-next=protected-access
            DragonConnector._send_req_with_socket(
                server_socket, DragonShutdownRequest()
            )
    except (zmq.error.ZMQError, zmq.Again) as e:
        # Can't use the logger as I/O file may be closed
        print("Could not send shutdown request to dragon server")
        print(f"ZMQ error: {e}", flush=True)
    finally:
        print("Sending shutdown request is complete")

    try:
        if server_authenticator is not None and server_authenticator.is_alive():
            print("Shutting down ZMQ authenticator")
            server_authenticator.stop()
    except Exception:
        print("Authenticator shutdown error")
    finally:
        print("Authenticator shutdown is complete")

    if server_process_pid and psutil.pid_exists(server_process_pid):
        print("Sending SIGINT to dragon server")
        retcode = None
        try:
            if graceful:
                os.kill(server_process_pid, signal.SIGINT)
                time.sleep(2)
                os.kill(server_process_pid, signal.SIGINT)
                time.sleep(10)
            os.kill(server_process_pid, signal.SIGTERM)
            _, retcode = os.waitpid(server_process_pid, 0)
        except ProcessLookupError:
            # Can't use the logger as I/O file may be closed
            print("Dragon server is not running.")
        finally:
            print(
                f"Dragon server process shutdown is complete , return code {retcode}",
                flush=True,
            )


def _resolve_dragon_path(fallback: t.Union[str, "os.PathLike[str]"]) -> Path:
    dragon_server_path = CONFIG.dragon_server_path or os.path.join(
        fallback, ".smartsim", "dragon"
    )
    dragon_server_paths = dragon_server_path.split(":")
    if len(dragon_server_paths) > 1:
        logger.warning(
            "Multiple dragon servers not supported, "
            "will connect to (or start) first server in list."
        )
    return Path(dragon_server_paths[0])
