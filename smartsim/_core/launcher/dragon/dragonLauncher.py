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

import zmq

from smartsim._core.launcher.dragon import dragonSockets
from smartsim.error.errors import SmartSimError

from ....error import LauncherError
from ....log import get_logger
from ....settings import DragonRunSettings, RunSettings, SettingsBase
from ....status import SmartSimStatus
from ...config import CONFIG
from ...schemas import (
    DragonBootstrapRequest,
    DragonBootstrapResponse,
    DragonHandshakeRequest,
    DragonHandshakeResponse,
    DragonRequest,
    DragonResponse,
    DragonRunRequest,
    DragonRunResponse,
    DragonShutdownRequest,
    DragonStopRequest,
    DragonStopResponse,
    DragonUpdateStatusRequest,
    DragonUpdateStatusResponse,
)
from ...utils.network import find_free_port, get_best_interface_and_address
from ..launcher import WLMLauncher
from ..step import DragonStep, LocalStep, Step
from ..stepInfo import StepInfo

logger = get_logger(__name__)

_SchemaT = t.TypeVar("_SchemaT", bound=t.Union[DragonRequest, DragonResponse])

DRG_LOCK = RLock()
DRG_CTX = zmq.Context()
DRG_CTX.setsockopt(zmq.REQ_CORRELATE, 1)
DRG_CTX.setsockopt(zmq.REQ_RELAXED, 1)


class DragonLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use Dragon on top of a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case Dragon. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object
    """

    def __init__(self) -> None:
        super().__init__()
        self._context = DRG_CTX
        self._timeout = CONFIG.dragon_server_timeout
        self._reconnect_timeout = CONFIG.dragon_server_reconnect_timeout
        self._startup_timeout = CONFIG.dragon_server_startup_timeout
        self._context.setsockopt(zmq.SNDTIMEO, value=self._timeout)
        self._context.setsockopt(zmq.RCVTIMEO, value=self._timeout)
        self._dragon_head_socket: t.Optional[zmq.Socket[t.Any]] = None
        self._dragon_head_process: t.Optional[subprocess.Popen[bytes]] = None
        # Returned by dragon head, useful if shutdown is to be requested
        # but process was started by another launcher
        self._dragon_head_pid: t.Optional[int] = None
        self._dragon_server_path = os.getenv(
            "SMARTSIM_DRAGON_SERVER_PATH_EXP",
            os.getenv("SMARTSIM_DRAGON_SERVER_PATH", None),
        )
        if self._dragon_server_path is None:
            raise SmartSimError(
                "Dragon server path was not set. "
                "This should not happen if the launcher was started by an experiment.\n"
                "If the DragonLauncher was started manually, "
                "then the environment variable SMARTSIM_DRAGON_SERVER_PATH "
                "should be set to an existing directory."
            )

    @property
    def is_connected(self) -> bool:
        return self._dragon_head_socket is not None

    def _handshake(self, address: str) -> None:
        self._dragon_head_socket = self._context.socket(zmq.REQ)
        self._dragon_head_socket.connect(address)
        try:
            dragon_handshake = _assert_schema_type(
                self._send_request(DragonHandshakeRequest()), DragonHandshakeResponse
            )
            self._dragon_head_pid = dragon_handshake.dragon_pid
            logger.debug(
                f"Successful handshake with Dragon server at address {address}"
            )
        except (zmq.ZMQError, zmq.Again) as e:
            logger.debug(e)
            self._dragon_head_socket.close()
            self._dragon_head_socket = None
            raise LauncherError(
                f"Unsuccessful handshake with Dragon server at address {address}"
            ) from e

    def _set_timeout(self, timeout: int) -> None:
        self._context.setsockopt(zmq.SNDTIMEO, value=timeout)
        self._context.setsockopt(zmq.RCVTIMEO, value=timeout)

    def ensure_connected(self) -> None:
        if not self.is_connected:
            self._connect_to_dragon()
        if not self.is_connected:
            raise LauncherError("Could not connect to Dragon server")

    # pylint: disable-next=too-many-statements,too-many-locals
    def _connect_to_dragon(self) -> None:
        with DRG_LOCK:
            # TODO use manager instead
            if self.is_connected:
                return

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
                        self._set_timeout(self._reconnect_timeout)
                        self._handshake(dragon_conf["address"])
                    except LauncherError as e:
                        logger.warning(e)
                    finally:
                        self._set_timeout(self._timeout)
                    if self.is_connected:
                        return

            path.mkdir(parents=True, exist_ok=True)

            cmd = [
                "dragon",
                sys.executable,
                "-m",
                "smartsim._core.entrypoints.dragon",
            ]

            address = get_best_interface_and_address().address
            socket_addr = ""
            launcher_socket: t.Optional[zmq.Socket[t.Any]] = None
            if address is not None:
                self._set_timeout(self._startup_timeout)
                launcher_socket = self._context.socket(zmq.REP)

                # find first available port >= 5995
                port = find_free_port(start=5995)
                socket_addr = f"tcp://{address}:{port}"
                logger.debug(f"Binding launcher to {socket_addr}")

                launcher_socket.bind(socket_addr)
                cmd += ["+launching_address", socket_addr]

            dragon_out_file = path / "dragon_head.out"
            dragon_err_file = path / "dragon_head.err"

            with open(dragon_out_file, "w", encoding="utf-8") as dragon_out, open(
                dragon_err_file, "w", encoding="utf-8"
            ) as dragon_err:
                current_env = os.environ.copy()
                current_env.update({"PYTHONUNBUFFERED": "1"})
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

            if launcher_socket is None:
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
                server = dragonSockets.as_server(launcher_socket)
                logger.debug(f"Listening to {socket_addr}")
                request = _assert_schema_type(server.recv(), DragonBootstrapRequest)

                logger.debug(f"Connecting launcher to {request.address}")
                server.send(
                    DragonBootstrapResponse(dragon_pid=self._dragon_head_process.pid)
                )

                launcher_socket.close()
                self._set_timeout(self._timeout)
                self._handshake(request.address)

                # Only the launcher which started the server is
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
                    )
            else:
                # TODO parse output file
                log_dragon_outputs()
                raise LauncherError("Could not receive address of Dragon head process")

    def cleanup(self) -> None:
        if self._dragon_head_socket is not None and self._dragon_head_pid is not None:
            _dragon_cleanup(
                server_socket=self._dragon_head_socket,
                server_process_pid=self._dragon_head_pid,
            )

    # RunSettings types supported by this launcher
    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {DragonRunSettings: DragonStep, RunSettings: LocalStep}

    def run(self, step: Step) -> t.Optional[str]:
        """Run a job step through Slurm

        :param step: a job step instance
        :type step: Step
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        :rtype: str
        """

        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        step_id = None
        task_id = None

        cmd = step.get_launch_cmd()
        out, err = step.get_output_files()

        if isinstance(step, DragonStep):
            self.ensure_connected()
            run_args = step.run_settings.run_args
            env = step.run_settings.env_vars
            nodes = int(run_args.get("nodes", None) or 1)
            tasks_per_node = int(run_args.get("tasks-per-node", None) or 1)
            response = _assert_schema_type(
                self._send_request(
                    DragonRunRequest(
                        exe=cmd[0],
                        exe_args=cmd[1:],
                        path=step.cwd,
                        name=step.name,
                        nodes=nodes,
                        tasks_per_node=tasks_per_node,
                        env=env,
                        current_env=os.environ,
                        output_file=out,
                        error_file=err,
                    )
                ),
                DragonRunResponse,
            )
            step_id = task_id = str(response.step_id)
        else:
            # pylint: disable-next=consider-using-with
            out_strm = open(out, "w+", encoding="utf-8")
            # pylint: disable-next=consider-using-with
            err_strm = open(err, "w+", encoding="utf-8")
            task_id = self.task_manager.start_task(
                cmd, step.cwd, step.env, out=out_strm.fileno(), err=err_strm.fileno()
            )

        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        return step_id

    def stop(self, step_name: str) -> StepInfo:
        """Step a job step

        :param step_name: name of the job to stop
        :type step_name: str
        :return: update for job due to cancel
        :rtype: StepInfo
        """

        self.ensure_connected()

        stepmap = self.step_mapping[step_name]
        step_id = str(stepmap.step_id)
        _assert_schema_type(
            self._send_request(DragonStopRequest(step_id=step_id)), DragonStopResponse
        )

        _, step_info = self.get_step_update([step_name])[0]
        if not step_info:
            raise LauncherError(f"Could not get step_info for job step {step_name}")

        step_info.status = (
            SmartSimStatus.STATUS_CANCELLED  # set status to cancelled instead of failed
        )
        return step_info

    def _get_managed_step_update(self, step_ids: t.List[str]) -> t.List[StepInfo]:
        """Get step updates for Dragon-managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """

        self.ensure_connected()

        response = _assert_schema_type(
            self._send_request(DragonUpdateStatusRequest(step_ids=step_ids)),
            DragonUpdateStatusResponse,
        )

        # create StepInfo objects to return
        updates: t.List[StepInfo] = []
        # Order matters as we return an ordered list of StepInfo objects
        for step_id in step_ids:
            if step_id not in response.statuses:
                msg = "Missing step id update from Dragon launcher."
                if response.error_message is not None:
                    msg += "\nDragon backend reported following error: "
                    msg += response.error_message
                raise LauncherError(msg)

            status, ret_codes = response.statuses[step_id]
            if ret_codes:
                grp_ret_code = min(ret_codes)
                if any(ret_codes):
                    _err_msg = (
                        f"One or more processes failed for job {step_id}"
                        f"Return codes were: {ret_codes}"
                    )
                    logger.error(_err_msg)
            else:
                grp_ret_code = None
            info = StepInfo(status, str(status), grp_ret_code)

            updates.append(info)
        return updates

    def _send_request(self, request: DragonRequest, flags: int = 0) -> DragonResponse:
        if (socket := self._dragon_head_socket) is None:
            raise LauncherError("Launcher is not connected to Dragon")
        return self.send_req_with_socket(socket, request, flags)

    def __str__(self) -> str:
        return "Dragon"

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
    def send_req_with_socket(
        socket: zmq.Socket[t.Any], request: DragonRequest, flags: int = 0
    ) -> DragonResponse:
        client = dragonSockets.as_client(socket)
        with DRG_LOCK:
            logger.debug(f"Sending {type(request).__name__}: {request}")
            send_trials = 5
            while send_trials:
                try:
                    client.send(request, flags)
                    break
                except zmq.Again as e:
                    send_trials -= 1
                    logger.debug(
                        f"Could not send request in {client.socket.getsockopt(zmq.SNDTIMEO)/1000} seconds"
                    )
                    if send_trials < 1:
                        raise e

            time.sleep(1)
            receive_trials = 5
            while receive_trials:
                try:
                    response = client.recv()
                    break
                except zmq.Again as e:
                    receive_trials -= 1
                    logger.debug(
                        f"Did not receive response in {client.socket.getsockopt(zmq.RCVTIMEO)/1000} seconds"
                    )
                    if receive_trials < 1:
                        raise e

            logger.debug(f"Received {type(response).__name__}: {response}")
            return response


def _assert_schema_type(obj: object, typ: t.Type[_SchemaT], /) -> _SchemaT:
    if not isinstance(obj, typ):
        raise TypeError(f"Expected schema of type `{typ}`, but got {type(obj)}")
    return obj


def _dragon_cleanup(server_socket: zmq.Socket[t.Any], server_process_pid: int) -> None:
    try:
        DragonLauncher.send_req_with_socket(server_socket, DragonShutdownRequest())
    except zmq.error.ZMQError as e:
        # Can't use the logger as I/O file may be closed
        print("Could not send shutdown request to dragon server")
        print(f"ZMQ error: {e}", flush=True)
    finally:
        time.sleep(1)
        try:
            os.kill(server_process_pid, signal.SIGINT)
            print("Sent SIGINT to dragon server")
        except ProcessLookupError:
            # Can't use the logger as I/O file may be closed
            print("Dragon server is not running.", flush=True)


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
