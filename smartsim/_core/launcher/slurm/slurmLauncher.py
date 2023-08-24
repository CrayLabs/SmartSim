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

import os
import time
import typing as t
from shutil import which

from ....error import LauncherError
from ....log import get_logger
from ....settings import (
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    RunSettings,
    SbatchSettings,
    SettingsBase,
    SrunSettings,
)
from ....status import STATUS_CANCELLED
from ...config import CONFIG
from ..launcher import WLMLauncher
from ..step import (
    LocalStep,
    MpiexecStep,
    MpirunStep,
    OrterunStep,
    SbatchStep,
    SrunStep,
    Step,
)
from ..stepInfo import SlurmStepInfo, StepInfo
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
    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        # RunSettings types supported by this launcher
        return {
            SrunSettings: SrunStep,
            SbatchSettings: SbatchStep,
            MpirunSettings: MpirunStep,
            MpiexecSettings: MpiexecStep,
            OrterunSettings: OrterunStep,
            RunSettings: LocalStep,
        }

    def get_step_nodes(self, step_names: t.List[str]) -> t.List[t.List[str]]:
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
        step_str = _create_step_id_str([val for val in step_ids if val is not None])
        output, error = sstat([step_str, "-i", "-n", "-p", "-a"])

        if "error:" in error.split(" "):
            raise LauncherError("Failed to retrieve nodelist from stat")

        # parse node list for each step
        node_lists = []
        for step_id in step_ids:
            node_lists.append(parse_sstat_nodes(output, step_id or ""))

        if len(node_lists) < 1:
            raise LauncherError("Failed to retrieve nodelist from stat")
        return node_lists

    def run(self, step: Step) -> t.Optional[str]:
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
            return_code, out, err = self.task_manager.start_and_wait(cmd_list, step.cwd)
            if return_code != 0:
                raise LauncherError(f"Sbatch submission failed\n {out}\n {err}")
            if out:
                step_id = out.strip()
                logger.debug(f"Gleaned batch job id: {step_id} for {step.name}")

        # Launch a in-allocation or on-allocation (if srun) command
        elif isinstance(step, SrunStep):
            task_id = self.task_manager.start_task(cmd_list, step.cwd)
        else:
            # MPI/local steps don't direct output like slurm steps
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

        if not step_id and step.managed:
            step_id = self._get_slurm_step_id(step)

        self.step_mapping.add(step.name, step_id, task_id, step.managed)

        # give slurm a rest
        # TODO make this configurable
        time.sleep(1)

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
            step_id = str(stepmap.step_id)
            # Check if step_id is part of colon-separated run,
            # this is reflected in a '+' in the step id,
            # so that the format becomes 12345+1.0.
            # If we find it it can mean two things:
            # a MPMD srun command, or a heterogeneous job.
            # If it is a MPMD srun, then stop parent step because
            # sub-steps cannot be stopped singularly.
            sub_step = "+" in step_id
            het_job = os.getenv("SLURM_HET_SIZE") is not None
            # If it is a het job, we can stop
            # them like this. Slurm will throw an error, but
            # will actually kill steps correctly.
            if sub_step and not het_job:
                step_id = step_id.split("+", maxsplit=1)[0]
            scancel_rc, _, err = scancel([step_id])
            if scancel_rc != 0:
                if het_job:
                    msg = (
                        "SmartSim received a non-zero exit code while canceling"
                        f" a heterogeneous job step {step_name}!\n"
                        "The following error might be internal to Slurm\n"
                        "and the heterogeneous job step could have been correctly"
                        " canceled.\nSmartSim will consider it canceled.\n"
                    )
                else:
                    msg = f"Unable to cancel job step {step_name}\n{err}"
                logger.warning(msg)
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
    def _get_slurm_step_id(step: Step, interval: int = 2) -> str:
        """Get the step_id of a step from sacct

        Parses sacct output by looking for the step name
        e.g. the following

        SmartSim|119225|
        extern|119225.extern|
        m1-119225.0|119225.0|
        m2-119225.1|119225.1|
        """
        time.sleep(interval)
        step_id: t.Optional[str] = None
        trials = CONFIG.wlm_trials
        while trials > 0:
            output, err = sacct(["--noheader", "-p", "--format=jobname,jobid"])
            if err:
                logger.warning(f"An error occurred while calling sacct: {err}")

            step_id = parse_step_id_from_sacct(output, step.name)
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
        step_str = _create_step_id_str(step_ids)
        sacct_out, _ = sacct(["--noheader", "-p", "-b", "--jobs", step_str])
        # (status, returncode)
        stat_tuples = [parse_sacct(sacct_out, step_id) for step_id in step_ids]

        # create SlurmStepInfo objects to return
        updates: t.List[StepInfo] = []
        for stat_tuple, step_id in zip(stat_tuples, step_ids):
            _rc = int(stat_tuple[1]) if stat_tuple[1] else None
            info = SlurmStepInfo(stat_tuple[0], _rc)

            task_id = self.step_mapping.get_task_id(step_id)
            if task_id:
                # we still check the task manager for jobs that didn't ever
                # become a fully managed job (e.g. error in slurm arguments)
                tid = str(task_id)
                _, return_code, out, err = self.task_manager.get_task_update(tid)
                if return_code and return_code != 0:
                    # tack on Popen error and output to status update.
                    info.output = out
                    info.error = err

            updates.append(info)
        return updates

    @staticmethod
    def check_for_slurm() -> None:
        """Check if slurm is available

        This function checks for slurm commands where the experiment
        is bring run

        :raises LauncherError: if no access to slurm
        """
        if not which("sbatch") and not which("sacct"):
            error = (
                "User attempted Slurm methods without access to Slurm "
                "at the call site.\n"
            )
            raise LauncherError(error)

    def __str__(self) -> str:
        return "Slurm"


def _create_step_id_str(step_ids: t.List[str]) -> str:
    return ",".join(step_ids)
