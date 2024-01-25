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

import fileinput
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import typing as t
from pathlib import Path

import zmq

from ....error import LauncherError
from ....log import get_logger
from ....settings import DragonRunSettings, SettingsBase
from ....status import STATUS_CANCELLED
from ...schemas import (
    DragonHandshakeRequest,
    DragonHandshakeResponse,
    DragonRequest,
    DragonRunResponse,
    DragonStopRequest,
    DragonStopResponse,
    DragonUpdateStatusRequest,
    DragonUpdateStatusResponse,
)
from ...utils.network import get_best_interface_and_address
from ..launcher import WLMLauncher
from ..step import DragonStep, Step
from ..stepInfo import StepInfo

logger = get_logger(__name__)


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
        self.context = zmq.Context()
        self._dragon_head_socket: t.Optional[zmq.Socket[t.Any]] = None

    def _handsake(self, address: str) -> None:

        self._dragon_head_socket = self.context.socket(zmq.REQ)
        self._dragon_head_socket.connect(address)
        request = DragonHandshakeRequest()
        try:
            self._send_request_as_json(request)
            response = self._dragon_head_socket.recv_json()
            if response is not None:
                response = t.cast(str, response)
                DragonHandshakeResponse.model_validate(t.cast(str, json.loads(response)))
            return
        except (zmq.ZMQError, zmq.Again) as e:
            logger.debug(e)
            self._dragon_head_socket.close()
            self._dragon_head_socket = None
            logger.debug(f"Unsuccessful handshake with Dragon server at address {address}")

    def connect_to_dragon(self, path: str) -> None:
        # TODO use manager instead
        if self._dragon_head_socket is not None:
            return

        dragon_path = os.path.join(path, ".smartsim", "dragon")
        dragon_config_log = os.path.join(dragon_path, "dragon_config.log")

        if Path.is_file(Path(dragon_config_log)):
            dragon_confs = DragonLauncher._parse_launched_dragon_server_info_from_files(
                [dragon_config_log]
            )
            logger.debug(dragon_confs)
            for dragon_conf in dragon_confs:
                self._handsake(dragon_conf["address"])
                if self._dragon_head_socket is not None:
                    return

        os.makedirs(dragon_path, exist_ok=True)

        cmd = [
            "dragon",
            sys.executable,
            "-m",
            "smartsim._core.entrypoints.dragon",
        ]

        _, address = get_best_interface_and_address()
        if address is not None:
            launcher_socket = self.context.socket(zmq.REP)
            # TODO find first available port >= 5995
            socket_addr = f"tcp://{address}:5995"
            logger.debug(f"Binding launcher to {socket_addr}")
            launcher_socket.bind(socket_addr)
            cmd += ["+launching_address", socket_addr]

        dragon_out_file = os.path.join(dragon_path, "dragon_head.out")
        dragon_err_file = os.path.join(dragon_path, "dragon_hear.err")

        with open(dragon_out_file, "w") as dragon_out, open(dragon_err_file, "w") as dragon_err:
            current_env = os.environ.copy()
            current_env.update({"PYTHONUNBUFFERED": "1"})
            self._dragon_head_process = subprocess.Popen(
                args = cmd,
                bufsize = 0,
                stderr=dragon_err.fileno(),
                stdout=dragon_out.fileno(),
                cwd=dragon_path,
                shell=False,
                env=current_env,
                start_new_session=True
            )

        if address is not None:
            logger.debug(f"Listening to {socket_addr}")
            dragon_head_address = launcher_socket.recv_string()
            logger.debug(f"Connecting launcher to {dragon_head_address}")
            launcher_socket.send(b"ACK")
            launcher_socket.close()
            self._handsake(dragon_head_address)
        else:
            # TODO parse output file
            raise LauncherError("Could not receive address of Dragon head process")

    # RunSettings types supported by this launcher
    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {DragonRunSettings: DragonStep}

    def run(self, step: Step) -> t.Optional[str]:
        """Run a job step through Slurm

        :param step: a job step instance
        :type step: Step
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        :rtype: str
        """

        if self._dragon_head_socket is None:
            raise LauncherError("Dragon environment not connected")

        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        step_id = None
        task_id = None

        if isinstance(step, DragonStep):
            req = step.get_launch_request()
            self._send_request_as_json(req)
            response = DragonRunResponse.model_validate(
                json.loads(t.cast(str, self._dragon_head_socket.recv_json()))
            )
            step_id = response.step_id

        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        return step_id

    def stop(self, step_name: str) -> StepInfo:
        """Step a job step

        :param step_name: name of the job to stop
        :type step_name: str
        :return: update for job due to cancel
        :rtype: StepInfo
        """

        if self._dragon_head_socket is None:
            raise LauncherError("Launcher is not connected to Dragon.")

        stepmap = self.step_mapping[step_name]

        step_id = str(stepmap.step_id)
        request = DragonStopRequest(step_id=step_id)
        self._send_request_as_json(request)


        DragonStopResponse.model_validate(
            json.loads(t.cast(str, self._dragon_head_socket.recv_json()))
        )

        _, step_info = self.get_step_update([step_name])[0]
        if not step_info:
            raise LauncherError(f"Could not get step_info for job step {step_name}")

        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    def _get_managed_step_update(self, step_ids: t.List[str]) -> t.List[StepInfo]:
        """Get step updates for Dragon-managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """

        if self._dragon_head_socket is None:
            raise LauncherError("Launcher is not connected to Dragon.")

        request = DragonUpdateStatusRequest(step_ids=step_ids)
        self._send_request_as_json(request)

        response = DragonUpdateStatusResponse.model_validate(
            json.loads(t.cast(str,self._dragon_head_socket.recv_json()))
        )

        # create SlurmStepInfo objects to return
        updates: t.List[StepInfo] = []
        # Order matters as we return an ordered list of StepInfo objects
        for step_id in step_ids:
            if step_id not in response.statuses:
                msg = "Missing step id update from Dragon launcher"
                if response.error_message is not None:
                    msg += f"Dragon backend reported following error: {response.error_message}"
                raise LauncherError(msg)
            stat_tuple = response.statuses[step_id]
            info = StepInfo(stat_tuple[0], stat_tuple[0], stat_tuple[1])

            updates.append(info)
        return updates

    def _send_request_as_json(self, request: DragonRequest, **kwargs) -> None:
        if self._dragon_head_socket is None:
            raise LauncherError("Launcher is not connected to Dragon")
        req_json = request.model_dump_json()
        logger.debug(f"Sending request: {req_json}")
        self._dragon_head_socket.send_json(req_json, **kwargs)

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
        cls, file_paths: t.List[str], num_dragon_envs: t.Optional[int] = None
    ) -> t.List[t.Dict[str, str]]:
        file_copies = [(Path(file).parent / (Path(file).name+".copy")) for file in file_paths]
        for file, file_copy in zip(file_paths, file_copies):
            shutil.copyfile(file, file_copy)
        with fileinput.FileInput(file_copies) as ifstream:
            dragon_envs = cls._parse_launched_dragon_server_info_from_iterable(
                ifstream, num_dragon_envs
            )
            for file_copy in file_copies:
                os.remove(file_copy)
            return dragon_envs