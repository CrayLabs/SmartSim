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

import psutil

from smartsim._core.launcher.step import Step
from smartsim.settings import (
    RunSettings,
    SettingsBase,
    AprunSettings,
    CobaltBatchSettings,
    MpirunSettings,
    MpiexecSettings,
    OrterunSettings,
)

from ....error import LauncherError
from ....log import get_logger
from ....status import STATUS_CANCELLED, STATUS_COMPLETED
from ...config import CONFIG
from ..launcher import WLMLauncher
from ..pbs.pbsCommands import qdel, qstat
from ..step import (
    Step,
    AprunStep,
    CobaltBatchStep,
    LocalStep,
    MpiexecStep,
    MpirunStep,
    OrterunStep,
)
from ..stepInfo import CobaltStepInfo, StepInfo
from .cobaltParser import parse_cobalt_step_id, parse_cobalt_step_status, parse_qsub_out

logger = get_logger(__name__)


class CobaltLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use Cobalt as a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case Cobalt. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object
    """

    def __init__(self) -> None:
        super().__init__()
        self.user = psutil.Process().username()

    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {
            AprunSettings: AprunStep,
            CobaltBatchSettings: CobaltBatchStep,
            MpirunSettings: MpirunStep,
            MpiexecSettings: MpiexecStep,
            OrterunSettings: OrterunStep,
            RunSettings: LocalStep,
        }

    def run(self, step: Step) -> t.Optional[str]:
        """Run a job step through Cobalt

        :param step: a job step instance
        :type step: Step
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        :rtype: str
        """
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        cmd_list = step.get_launch_cmd()
        step_id = None
        task_id = None
        if isinstance(step, CobaltBatchStep):
            # wait for batch step to submit successfully
            return_code, out, err = self.task_manager.start_and_wait(cmd_list, step.cwd)
            if return_code != 0:
                raise LauncherError(
                    f"Cobalt qsub batch submission failed\n {out}\n {err}"
                )
            if out:
                step_id = parse_qsub_out(out)
                logger.debug(f"Gleaned batch job id: {step_id} for {step.name}")
        else:
            # aprun doesn't direct output for us.
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
            step_id = self._get_cobalt_step_id(step)

        self.step_mapping.add(step.name, step_id, task_id, step.managed)
        return step_id

    def stop(self, step_name: str) -> StepInfo:
        """Step a job step

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
            if stepmap.task_id:
                self.task_manager.remove_task(str(stepmap.task_id))

        _, step_info = self.get_step_update([step_name])[0]
        if not step_info:
            raise LauncherError(f"Could not get step_info for job step {step_name}")
        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    def _get_cobalt_step_id(self, step: Step, interval: int = 2) -> str:
        """Get the step_id of a step from qstat (rarely used)

        Parses cobalt qstat output by looking for the step name
        """
        step_id = None
        trials = CONFIG.wlm_trials
        while trials > 0:
            output, _ = qstat(["--header", "JobName:JobId", "-u", self.user])
            step_id = parse_cobalt_step_id(output, step.name)
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
        args = ["--header", "JobId:State", "-u", self.user]
        args.extend(step_ids)
        qstat_out, _ = qstat(args)

        stats = [
            parse_cobalt_step_status(qstat_out, str(step_id)) for step_id in step_ids
        ]
        # create CobaltStepInfo objects to return
        updates: t.List[StepInfo] = []
        for stat, _ in zip(stats, step_ids):
            info = CobaltStepInfo(stat, None)  # returncode not logged by Cobalt

            if info.status == STATUS_COMPLETED:
                info.returncode = 0

            updates.append(info)
        return updates

    def __str__(self) -> str:
        return "Cobalt"
