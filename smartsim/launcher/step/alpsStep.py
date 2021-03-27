import os
import os.path as osp
from itertools import product
from re import I
from ...error import SSConfigError
from .step import Step

from ...utils import get_logger
logger = get_logger(__name__)


class AprunStep(Step):

    def __init__(self, name, cwd, run_settings):
        """Initialize a ALPS aprun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        """
        super().__init__(name, cwd)
        self.run_settings = run_settings
        self.alloc = None
        if not self.run_settings.in_batch:
            self._set_alloc()

    def get_launch_cmd(self):
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        aprun = self.run_settings.run_command
        aprun_cmd = [aprun, "--wdir", self.cwd]
        aprun_cmd += self.run_settings.format_env_vars()
        aprun_cmd += self._build_exe()

        # if its in a batch, redirect stdout to
        # file in the cwd.
        if self.run_settings.in_batch:
            output = self.get_step_file(ending=".out")
            aprun_cmd += [">", output]
        return aprun_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises SSConfigError: allocation not listed or found
        """
        if "PBS_JOBID" in os.environ:
            self.alloc = os.environ["PBS_JOBID"]
            logger.debug(
                f"Running on PBS allocation {self.alloc} gleaned from user environment")
        if "COBALT_JOBID" in os.environ:
            self.alloc = os.environ["COBALT_JOBID"]
            logger.debug(
                f"Running on Cobalt allocation {self.alloc} gleaned from user environment")
        else:
            raise SSConfigError(
                "No allocation specified or found and not running in batch")

    def _build_exe(self):
        """Build the executable for this step

        :return: executable list
        :rtype: list[str]
        """
        if self.run_settings.mpmd:
           return self._make_mpmd()
        else:
            cmd = self.run_settings.format_run_args()
            cmd += self.run_settings.exe
            cmd += self.run_settings.exe_args
            return cmd

    def _make_mpmd(self):
        """Build Aprun (MPMD) executable
        """
        cmd = self.run_settings.format_run_args()
        cmd += self.run_settings.exe
        cmd += self.run_settings.exe_args
        for mpmd in self.run_settings.mpmd:
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += mpmd.exe
            cmd += mpmd.exe_args
        return cmd

