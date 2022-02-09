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
from shutil import which

from ....error import LauncherError
from ....log import get_logger
from ....settings import *
from ....status import STATUS_CANCELLED
from ..launcher import WLMLauncher
from ..step import LocalStep, MpirunStep, SbatchStep, SrunStep
from ..stepInfo import SlurmStepInfo
from .slurmCommands import sacct, scancel, sstat
from .slurmParser import parse_sacct, parse_sstat_nodes, parse_step_id_from_sacct

logger = get_logger(__name__)


class SlurmLauncher(WLMLauncher):
    """This class encapsulates the functionality needed
    to launch jobs on systems that use Slurm as a workload manager.

    All WLM launchers are capable of launching managed and unmanaged
    jobs. Managed jobs are queried through interaction with with WLM,
    in this case Slurm. Unmanaged jobs are held in the TaskManager
    and are managed through references to their launching process ID
    i.e. a psutil.Popen object
    """

    # init in launcher.py (WLMLauncher)

    # RunSettings types supported by this launcher
    supported_rs = {
        SrunSettings: SrunStep,
        SbatchSettings: SbatchStep,
        MpirunSettings: MpirunStep,
        RunSettings: LocalStep,
    }

    def get_step_nodes(self, step_names):
        """Return the compute nodes of a specific job or allocation

        This function returns the compute nodes of a specific job or allocation
        in a list with the duplicates removed.

        Output gleaned from sstat e.g. the following

        29917893.extern|nid00034|44860|
        29917893.0|nid00034|44887,45151,45152,45153,45154,45155|
        29917893.2|nid00034|45174|

        would return nid00034

        :param step_names: list of job step names
        :type step_names: list[str]
        :raises LauncherError: if nodelist aquisition fails
        :return: list of hostnames
        :rtype: list[str]
        """
        _, step_ids = self.step_mapping.get_ids(step_names, managed=True)
        step_str = _create_step_id_str(step_ids)
        output, error = sstat([step_str, "-i", "-n", "-p", "-a"])

        if "error:" in error.split(" "):
            raise LauncherError("Failed to retrieve nodelist from stat")

        # parse node list for each step
        node_lists = []
        for step_id in step_ids:
            node_lists.append(parse_sstat_nodes(output, step_id))

        if len(node_lists) < 1:
            raise LauncherError("Failed to retrieve nodelist from stat")
        return node_lists

    def run(self, step):
        """Run a job step through Slurm

        :param step: a job step instance
        :type step: Step
        :raises LauncherError: if launch fails
        :return: job step id if job is managed
        :rtype: str
        """
        self.check_for_slurm()
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        cmd_list = step.get_launch_cmd()
        step_id = None
        task_id = None

        # Launch a batch step with Slurm
        if isinstance(step, SbatchStep):
            # wait for batch step to submit successfully
            rc, out, err = self.task_manager.start_and_wait(cmd_list, step.cwd)
            if rc != 0:
                raise LauncherError(f"Sbatch submission failed\n {out}\n {err}")
            if out:
                step_id = out.strip()
                logger.debug(f"Gleaned batch job id: {step_id} for {step.name}")

        # Launch a in-allocation or on-allocation (if srun) command
        else:
            if isinstance(step, SrunStep):
                task_id = self.task_manager.start_task(cmd_list, step.cwd)
            else:
                # Mpirun doesn't direct output for us like srun does
                out, err = step.get_output_files()
                output = open(out, "w+")
                error = open(err, "w+")
                task_id = self.task_manager.start_task(
                    cmd_list, step.cwd, out=output, err=error
                )

        if not step_id and step.managed:
            step_id = self._get_slurm_step_id(step)
        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        # give slurm a rest
        # TODO make this configurable
        time.sleep(1)

        return step_id

    def stop(self, step_name):
        """Step a job step

        :param step_name: name of the job to stop
        :type step_name: str
        :return: update for job due to cancel
        :rtype: StepInfo
        """
        stepmap = self.step_mapping[step_name]
        if stepmap.managed:
            step_id = str(stepmap.step_id)
            # Check if step_id is part of colon-separated run
            # if that is the case, stop parent job step because
            # sub-steps cannot be stopped singularly.
            if "+" in step_id:
                step_id = step_id.split("+")[0]
            scancel_rc, _, err = scancel([step_id])
            if scancel_rc != 0:
                logger.warning(f"Unable to cancel job step {step_name}\n {err}")
            if stepmap.task_id:
                self.task_manager.remove_task(stepmap.task_id)
        else:
            self.task_manager.remove_task(stepmap.task_id)

        _, step_info = self.get_step_update([step_name])[0]
        step_info.status = STATUS_CANCELLED  # set status to cancelled instead of failed
        return step_info

    def _get_slurm_step_id(self, step, interval=2, trials=5):
        """Get the step_id of a step from sacct

        Parses sacct output by looking for the step name
        e.g. the following

        SmartSim|119225|
        extern|119225.extern|
        m1-119225.0|119225.0|
        m2-119225.1|119225.1|
        """
        time.sleep(interval)
        step_id = "unassigned"
        while trials > 0:
            output, _ = sacct(["--noheader", "-p", "--format=jobname,jobid"])
            step_id = parse_step_id_from_sacct(output, step.name)
            if step_id:
                break
            else:
                time.sleep(interval)
                trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return step_id

    def _get_managed_step_update(self, step_ids):
        """Get step updates for WLM managed jobs

        :param step_ids: list of job step ids
        :type step_ids: list[str]
        :return: list of updates for managed jobs
        :rtype: list[StepInfo]
        """
        step_str = _create_step_id_str(step_ids)
        sacct_out, _ = sacct(["--noheader", "-p", "-b", "--jobs", step_str])
        # (status, returncode)
        stat_tuples = [parse_sacct(sacct_out, step_id) for step_id in step_ids]

        # create SlurmStepInfo objects to return
        updates = []
        for stat_tuple, step_id in zip(stat_tuples, step_ids):
            info = SlurmStepInfo(stat_tuple[0], stat_tuple[1])

            task_id = self.step_mapping.get_task_id(step_id)
            if task_id:
                # we still check the task manager for jobs that didn't ever
                # become a fully managed job (e.g. error in slurm arguments)
                _, rc, out, err = self.task_manager.get_task_update(task_id)
                if rc and rc != 0:
                    # tack on Popen error and output to status update.
                    info.output = out
                    info.error = err

            updates.append(info)
        return updates

    @staticmethod
    def check_for_slurm():
        """Check if slurm is available

        This function checks for slurm commands where the experiment
        is bring run

        :raises LauncherError: if no access to slurm
        """
        if not which("sbatch") and not which("sacct"):
            error = "User attempted Slurm methods without access to Slurm at the call site.\n"
            raise LauncherError(error)

    def __str__(self):
        return "Slurm"


def _create_step_id_str(step_ids):
    step_str = ""
    for step_id in step_ids:
        step_str += str(step_id) + ","
    step_str = step_str.strip(",")
    return step_str
