# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from ....error import LauncherError
from ....log import get_logger
from ....settings import *
from ....status import STATUS_CANCELLED, STATUS_COMPLETED
from ..launcher import WLMLauncher
from ..step import BsubBatchStep, JsrunStep, LocalStep, MpirunStep
from ..stepInfo import LSFBatchStepInfo, LSFJsrunStepInfo
from .lsfCommands import bjobs, bkill, jskill, jslist
from .lsfParser import (
    parse_bjobs_jobid,
    parse_bsub,
    parse_jslist_stepid,
    parse_max_step_id_from_jslist,
)

logger = get_logger(__name__)


class LSFLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use LSF as a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case LSF. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object
    """

    # init in WLMLauncher, launcher.py

    # RunSettings types supported by this launcher
    supported_rs = {
        JsrunSettings: JsrunStep,
        BsubBatchSettings: BsubBatchStep,
        MpirunSettings: MpirunStep,
        RunSettings: LocalStep,
    }

    def run(self, step):
        """Run a job step through LSF

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
        if isinstance(step, BsubBatchStep):
            # wait for batch step to submit successfully
            rc, out, err = self.task_manager.start_and_wait(cmd_list, step.cwd)
            if rc != 0:
                raise LauncherError(f"Bsub batch submission failed\n {out}\n {err}")
            if out:
                step_id = parse_bsub(out)
                logger.debug(f"Gleaned batch job id: {step_id} for {step.name}")
        elif isinstance(step, JsrunStep):
            self.task_manager.start_task(cmd_list, step.cwd)
            time.sleep(1)
            step_id = self._get_lsf_step_id(step)
            logger.debug(f"Gleaned jsrun step id: {step_id} for {step.name}")
        else:  # isinstance(step, MpirunStep) or isinstance(step, LocalStep)
            out, err = step.get_output_files()
            # mpirun and local launch don't direct output for us
            output = open(out, "w+")
            error = open(err, "w+")
            task_id = self.task_manager.start_task(
                cmd_list, step.cwd, out=output, err=error
            )

        self.step_mapping.add(step.name, step_id, task_id, step.managed)
        return step_id

    def stop(self, step_name):
        """Stop/cancel a job step

        :param step_name: name of the job to stop
        :type step_name: str
        :return: update for job due to cancel
        :rtype: StepInfo
        """
        stepmap = self.step_mapping[step_name]
        if stepmap.managed:
            if "." in stepmap.step_id:
                rc, _, err = jskill([stepmap.step_id.rpartition(".")[-1]])
            else:
                rc, _, err = bkill([str(stepmap.step_id)])
            if rc != 0:
                logger.warning(f"Unable to cancel job step {step_name}\n {err}")
            if stepmap.task_id:
                self.task_manager.remove_task(stepmap.task_id)
        else:
            self.task_manager.remove_task(stepmap.task_id)

        _, step_info = self.get_step_update([step_name])[0]
        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    def _get_lsf_step_id(self, step, interval=2, trials=5):
        """Get the step_id of last launched step from jslist"""
        time.sleep(interval)
        step_id = "unassigned"
        while trials > 0:
            output, _ = jslist([])
            step_id = parse_max_step_id_from_jslist(output)
            if step_id:
                break
            else:
                time.sleep(interval)
                trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return f"{step.alloc}.{step_id}"

    def _get_managed_step_update(self, step_ids):
        """Get step updates for WLM managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """
        updates = []

        for step_id in step_ids:

            # Batch jobs have integer step id,
            # Jsrun processes have {alloc}.{task_id}
            # Include recently finished jobs
            if "." in str(step_id):
                jsrun_step_id = step_id.rpartition(".")[-1]
                jslist_out, _ = jslist([])
                stat, return_code = parse_jslist_stepid(jslist_out, jsrun_step_id)
                info = LSFJsrunStepInfo(stat, return_code)
            else:
                bjobs_args = ["-a"] + step_ids
                bjobs_out, _ = bjobs(bjobs_args)
                stat = parse_bjobs_jobid(bjobs_out, str(step_id))
                # create LSFBatchStepInfo objects to return
                info = LSFBatchStepInfo(stat, None)
                # account for case where job history is not logged by LSF
                if info.status == STATUS_COMPLETED:
                    info.returncode = 0

            updates.append(info)
        return updates

    def __str__(self):
        return "LSF"
