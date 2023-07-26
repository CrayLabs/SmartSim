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

import time
import typing as t

from ....error import LauncherError
from ....log import get_logger
from ....settings import (
    SettingsBase,
    AprunSettings,
    QsubBatchSettings,
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    RunSettings,
    PalsMpiexecSettings,
)
from ....status import STATUS_CANCELLED, STATUS_COMPLETED
from ...config import CONFIG
from ..launcher import WLMLauncher
from ..step import (
    Step,
    AprunStep,
    LocalStep,
    MpiexecStep,
    MpirunStep,
    OrterunStep,
    QsubBatchStep,
)
from ..stepInfo import PBSStepInfo, StepInfo
from .pbsCommands import qdel, qstat
from .pbsParser import parse_qstat_jobid, parse_step_id_from_qstat

logger = get_logger(__name__)


class PBSLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use PBS as a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case PBS. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object
    """

    # init in WLMLauncher, launcher.py

    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {
            AprunSettings: AprunStep,
            QsubBatchSettings: QsubBatchStep,
            MpiexecSettings: MpiexecStep,
            MpirunSettings: MpirunStep,
            OrterunSettings: OrterunStep,
            RunSettings: LocalStep,
            PalsMpiexecSettings: MpiexecStep,
        }

    def run(self, step: Step) -> t.Optional[str]:
        """Run a job step through PBSPro

        :param step: a job step instance
        :type step: Step
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        :rtype: str
        """
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        cmd_list = step.get_launch_cmd()
        step_id: t.Optional[str] = None
        task_id: t.Optional[str] = None
        if isinstance(step, QsubBatchStep):
            # wait for batch step to submit successfully
            return_code, out, err = self.task_manager.start_and_wait(cmd_list, step.cwd)
            if return_code != 0:
                raise LauncherError(f"Qsub batch submission failed\n {out}\n {err}")
            if out:
                step_id = out.strip()
                logger.debug(f"Gleaned batch job id: {step_id} for {step.name}")
        else:
            # aprun/local doesn't direct output for us.
            out, err = step.get_output_files()

            # LocalStep.run_command omits env, include it here
            passed_env = step.env if isinstance(step, LocalStep) else None

            # pylint: disable-next=consider-using-with
            output = open(out, "w+", encoding="utf-8")
            # pylint: disable-next=consider-using-with
            error = open(err, "w+", encoding="utf-8")
            task_id = self.task_manager.start_task(
                cmd_list, step.cwd, passed_env, out=output.fileno(), err=error.fileno()
            )

        # if batch submission did not successfully retrieve job ID
        if not step_id and step.managed:
            step_id = self._get_pbs_step_id(step)

        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        return step_id

    def stop(self, step_name: str) -> StepInfo:
        """Stop/cancel a job step

        :param step_name: name of the job to stop
        :type step_name: str
        :return: update for job due to cancel
        :rtype: StepInfo
        """
        stepmap = self.step_mapping[step_name]
        if stepmap.managed:
            qdel_rc, _, err = qdel([str(stepmap.step_id)])
            if qdel_rc != 0:
                logger.warning(f"Unable to cancel job step {step_name}\n {err}")
            if stepmap.task_id:
                self.task_manager.remove_task(str(stepmap.task_id))
        else:
            self.task_manager.remove_task(str(stepmap.task_id))

        _, step_info = self.get_step_update([step_name])[0]
        if not step_info:
            raise LauncherError(f"Could not get step_info for job step {step_name}")

        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    @staticmethod
    def _get_pbs_step_id(step: Step, interval: int = 2) -> str:
        """Get the step_id of a step from qstat (rarely used)

        Parses qstat JSON output by looking for the step name
        TODO: change this to use ``qstat -a -u user``
        """
        time.sleep(interval)
        step_id: t.Optional[str] = None
        trials = CONFIG.wlm_trials
        while trials > 0:
            output, _ = qstat(["-f", "-F", "json"])
            step_id = parse_step_id_from_qstat(output, step.name)
            if step_id:
                break
            else:
                time.sleep(interval)
                trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return step_id

    def _get_managed_step_update(self, step_ids: t.List[str]) -> t.List[StepInfo]:
        """Get step updates for WLM managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """
        updates: t.List[StepInfo] = []

        qstat_out, _ = qstat(step_ids)
        stats = [parse_qstat_jobid(qstat_out, str(step_id)) for step_id in step_ids]
        # create PBSStepInfo objects to return

        for stat, _ in zip(stats, step_ids):
            info = PBSStepInfo(stat, None)
            # account for case where job history is not logged by PBS
            if info.status == STATUS_COMPLETED:
                info.returncode = 0

            updates.append(info)
        return updates

    def __str__(self) -> str:
        return "PBS"
