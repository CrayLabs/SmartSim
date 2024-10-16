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

import os
import typing as t

from smartsim._core.schemas.dragonRequests import DragonRunPolicy

from ...._core.launcher.stepMapping import StepMap
from ....error import LauncherError, SmartSimError
from ....log import get_logger
from ....settings import (
    DragonRunSettings,
    QsubBatchSettings,
    RunSettings,
    SbatchSettings,
    SettingsBase,
)
from ....status import SmartSimStatus
from ...schemas import (
    DragonRunRequest,
    DragonRunResponse,
    DragonStopRequest,
    DragonStopResponse,
    DragonUpdateStatusRequest,
    DragonUpdateStatusResponse,
)
from ..launcher import WLMLauncher
from ..pbs.pbsLauncher import PBSLauncher
from ..slurm.slurmLauncher import SlurmLauncher
from ..step import DragonBatchStep, DragonStep, LocalStep, Step
from ..stepInfo import StepInfo
from .dragonConnector import DragonConnector, _SchemaT

logger = get_logger(__name__)


class DragonLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use Dragon on top of a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case the Dragon server. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object.
    Batch Jobs are routed to either Slurm or PBS and their step ID
    is stored, prefixed with the name of the scheduler, to allow
    the Job Manager to interact with it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._connector = DragonConnector()
        """Connector used to start and interact with the Dragon server"""
        self._slurm_launcher = SlurmLauncher()
        """Slurm sub-launcher, used only for batch jobs"""
        self._pbs_launcher = PBSLauncher()
        """PBS sub-launcher, used only for batch jobs"""

    @property
    def is_connected(self) -> bool:
        return self._connector.is_connected

    def cleanup(self) -> None:
        self._connector.cleanup()

    # RunSettings types supported by this launcher
    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {
            DragonRunSettings: DragonStep,
            SbatchSettings: DragonBatchStep,
            QsubBatchSettings: DragonBatchStep,
            RunSettings: LocalStep,
        }

    def add_step_to_mapping_table(self, name: str, step_map: StepMap) -> None:
        super().add_step_to_mapping_table(name, step_map)

        if step_map.step_id is None:
            return
        sublauncher: t.Optional[t.Union[SlurmLauncher, PBSLauncher]] = None
        if step_map.step_id.startswith("SLURM-"):
            sublauncher = self._slurm_launcher
        elif step_map.step_id.startswith("PBS-"):
            sublauncher = self._pbs_launcher
        else:
            return

        sublauncher_step_map = StepMap(
            step_id=DragonLauncher._unprefix_step_id(step_map.step_id),
            task_id=step_map.task_id,
            managed=step_map.managed,
        )
        sublauncher.add_step_to_mapping_table(name, sublauncher_step_map)

    def run(self, step: Step) -> t.Optional[str]:
        """Run a job step through Slurm

        :param step: a job step instance
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        """

        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        step_id = None
        task_id = None

        cmd = step.get_launch_cmd()
        out, err = step.get_output_files()

        if isinstance(step, DragonBatchStep):
            # wait for batch step to submit successfully
            sublauncher_step_id: t.Optional[str] = None
            return_code, out, err = self.task_manager.start_and_wait(cmd, step.cwd)
            if return_code != 0:
                raise LauncherError(f"Sbatch submission failed\n {out}\n {err}")
            if out:
                sublauncher_step_id = out.strip()
                logger.debug(
                    f"Gleaned batch job id: {sublauncher_step_id} for {step.name}"
                )

            if sublauncher_step_id is None:
                raise SmartSimError("Could not get step id for batch step")

            if isinstance(step.batch_settings, SbatchSettings):
                self._slurm_launcher.step_mapping.add(
                    step.name, sublauncher_step_id, task_id, step.managed
                )
                step_id = "SLURM-" + sublauncher_step_id
            elif isinstance(step.batch_settings, QsubBatchSettings):
                self._pbs_launcher.step_mapping.add(
                    step.name, sublauncher_step_id, task_id, step.managed
                )
                step_id = "PBS-" + sublauncher_step_id
        elif isinstance(step, DragonStep):
            run_args = step.run_settings.run_args
            req_env = step.run_settings.env_vars
            self._connector.load_persisted_env()
            merged_env = self._connector.merge_persisted_env(os.environ.copy())
            nodes = int(run_args.get("nodes", None) or 1)
            tasks_per_node = int(run_args.get("tasks-per-node", None) or 1)
            hosts = run_args.get("host-list", None)

            policy = DragonRunPolicy.from_run_args(run_args)

            response = _assert_schema_type(
                self._connector.send_request(
                    DragonRunRequest(
                        exe=cmd[0],
                        exe_args=cmd[1:],
                        path=step.cwd,
                        name=step.name,
                        nodes=nodes,
                        tasks_per_node=tasks_per_node,
                        env=req_env,
                        current_env=merged_env,
                        output_file=out,
                        error_file=err,
                        policy=policy,
                        hostlist=hosts,
                    )
                ),
                DragonRunResponse,
            )
            step_id = str(response.step_id)
        else:
            # pylint: disable-next=consider-using-with
            out_strm = open(out, "w+", encoding="utf-8")
            # pylint: disable-next=consider-using-with
            err_strm = open(err, "w+", encoding="utf-8")
            task_id = self.task_manager.start_task(
                cmd, step.cwd, step.env, out=out_strm.fileno(), err=err_strm.fileno()
            )
            step.managed = False

        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        return step_id

    def stop(self, step_name: str) -> StepInfo:
        """Step a job step

        :param step_name: name of the job to stop
        :return: update for job due to cancel
        """

        stepmap = self.step_mapping[step_name]
        step_id = str(stepmap.step_id)

        if step_id.startswith("SLURM-"):
            return self._slurm_launcher.stop(step_name)

        if step_id.startswith("PBS-"):
            return self._pbs_launcher.stop(step_name)

        _assert_schema_type(
            self._connector.send_request(DragonStopRequest(step_id=step_id)),
            DragonStopResponse,
        )

        _, step_info = self.get_step_update([step_name])[0]
        if not step_info:
            raise LauncherError(f"Could not get step_info for job step {step_name}")

        step_info.status = (
            SmartSimStatus.STATUS_CANCELLED  # set status to cancelled instead of failed
        )
        step_info.launcher_status = str(SmartSimStatus.STATUS_CANCELLED)
        return step_info

    @staticmethod
    def _unprefix_step_id(step_id: str) -> str:
        return step_id.split("-", maxsplit=1)[1]

    def _get_managed_step_update(self, step_ids: t.List[str]) -> t.List[StepInfo]:
        """Get step updates for Dragon-managed jobs

        :param step_ids: list of job step ids
        :return: list of updates for managed jobs
        """

        step_id_updates: dict[str, StepInfo] = {}

        dragon_step_ids: t.List[str] = []
        slurm_step_ids: t.List[str] = []
        pbs_step_ids: t.List[str] = []
        for step_id in step_ids:
            if step_id.startswith("SLURM-"):
                slurm_step_ids.append(step_id)
            elif step_id.startswith("PBS-"):
                pbs_step_ids.append(step_id)
            else:
                dragon_step_ids.append(step_id)

        if slurm_step_ids:
            # pylint: disable-next=protected-access
            slurm_updates = self._slurm_launcher._get_managed_step_update(
                [
                    DragonLauncher._unprefix_step_id(step_id)
                    for step_id in slurm_step_ids
                ]
            )
            step_id_updates.update(dict(zip(slurm_step_ids, slurm_updates)))

        if pbs_step_ids:
            # pylint: disable-next=protected-access
            pbs_updates = self._pbs_launcher._get_managed_step_update(
                [DragonLauncher._unprefix_step_id(step_id) for step_id in pbs_step_ids]
            )
            step_id_updates.update(dict(zip(pbs_step_ids, pbs_updates)))

        if dragon_step_ids:
            response = _assert_schema_type(
                self._connector.send_request(
                    DragonUpdateStatusRequest(step_ids=dragon_step_ids)
                ),
                DragonUpdateStatusResponse,
            )

            for step_id in step_ids:
                if step_id not in response.statuses:
                    msg = "Missing step id update from Dragon launcher."
                    if response.error_message is not None:
                        msg += "\nDragon backend reported following error: "
                        msg += response.error_message
                    logger.error(msg)
                    info = StepInfo(
                        SmartSimStatus.STATUS_FAILED,
                        SmartSimStatus.STATUS_FAILED.value,
                        -1,
                    )
                else:
                    status, ret_codes = response.statuses[step_id]
                    if ret_codes:
                        grp_ret_code = min(ret_codes)
                        if any(ret_codes):
                            _err_msg = (
                                f"One or more processes failed for job {step_id} "
                                f"Return codes were: {ret_codes}"
                            )
                            logger.error(_err_msg)
                    else:
                        grp_ret_code = None
                    info = StepInfo(status, status.value, grp_ret_code)

                step_id_updates[step_id] = info

        # Order matters as we return an ordered list of StepInfo objects
        return [step_id_updates[step_id] for step_id in step_ids]

    def __str__(self) -> str:
        return "Dragon"


def _assert_schema_type(obj: object, typ: t.Type[_SchemaT], /) -> _SchemaT:
    if not isinstance(obj, typ):
        raise TypeError(f"Expected schema of type `{typ}`, but got {type(obj)}")
    return obj
