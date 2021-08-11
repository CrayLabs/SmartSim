# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import psutil

from ...constants import STATUS_CANCELLED, STATUS_COMPLETED
from ...error import LauncherError, SSConfigError
from ...settings import BsubBatchSettings, JsrunSettings, MpirunSettings
from ...utils import get_logger
from ..launcher import WLMLauncher
from ..step import BsubBatchStep, JsrunStep, MpirunStep
from ..stepInfo import LSFStepInfo
from .lsfCommands import bjobs, bkill
from .lsfParser import parse_bjobs_jobid, parse_bsub, parse_step_id_from_bjobs

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

    def create_step(self, name, cwd, step_settings):
        """Create a LSF job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param step_settings: batch or run settings for entity
        :type step_settings: BatchSettings | RunSettings
        :raises SSUnsupportedError: if batch or run settings type isnt supported
        :raises LauncherError: if step creation fails
        :return: step instance
        :rtype: Step
        """
        try:
            if isinstance(step_settings, JsrunSettings):
                step = JsrunStep(name, cwd, step_settings)
                return step
            if isinstance(step_settings, BsubBatchSettings):
                step = BsubBatchStep(name, cwd, step_settings)
                return step
            if isinstance(step_settings, MpirunSettings):
                step = MpirunStep(name, cwd, step_settings)
                return step
            raise TypeError(
                f"RunSettings type {type(step_settings)} not supported by LSF"
            )
        except SSConfigError as e:
            raise LauncherError("Job step creation failed: " + str(e)) from None

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
        elif isinstance(step, MpirunStep):
            out, err = step.get_output_files()
            # mpirun doesn't direct output for us
            output = open(out, "w+")
            error = open(err, "w+")
            task_id = self.task_manager.start_task(
                cmd_list, step.cwd, out=output, err=error
            )
        else:
            task_id = self.task_manager.start_task(cmd_list, step.cwd)

        # if batch submission did not successfully retrieve job ID
        if not step_id and step.managed:  # pragma: no cover
            step_id = self._get_lsf_step_id(step)
        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        time.sleep(5)

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
            qdel_rc, _, err = bkill([str(stepmap.step_id)])
            if qdel_rc != 0:
                logger.warning(f"Unable to cancel job step {step_name}\n {err}")
            if stepmap.task_id:
                self.task_manager.remove_task(stepmap.task_id)
        else:
            self.task_manager.remove_task(stepmap.task_id)

        _, step_info = self.get_step_update([step_name])[0]
        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    # TODO: use jslist here if it is a JsrunStep
    # otherwise, this is only reached in a very rare case where a batch
    # job is submitted but no message is receieved
    # We exclude this from coverage
    def _get_lsf_step_id(self, step, interval=2, trials=5):  # pragma: no cover
        """Get the step_id of a step from bjobs (rarely used)

        Parses bjobs output by looking for the step name
        """
        time.sleep(interval)
        step_id = "unassigned"
        username = psutil.Process.username()
        while trials > 0:
            output, _ = bjobs(["-w", "-u", username])
            step_id = parse_step_id_from_bjobs(output, step.name)
            if step_id:
                break
            else:
                time.sleep(interval)
                trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return step_id

    # TODO: use jslist here if it is a JsrunStep
    def _get_managed_step_update(self, step_ids):
        """Get step updates for WLM managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """
        updates = []
        # Include recently finished jobs
        bjobs_args = ["-a"] + step_ids
        bjobs_out, _ = bjobs(bjobs_args)
        stats = [parse_bjobs_jobid(bjobs_out, str(step_id)) for step_id in step_ids]
        # create LSFStepInfo objects to return

        for stat, _ in zip(stats, step_ids):
            info = LSFStepInfo(stat, None)
            # account for case where job history is not logged by LSF
            if info.status == STATUS_COMPLETED:
                info.returncode = 0

            updates.append(info)
        return updates

    def __str__(self):
        # TODO get the version here
        return "LSF"
