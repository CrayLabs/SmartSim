import os
import stat
from .step import Step
from ...utils import get_logger
logger = get_logger(__name__)


class CobaltBatchStep(Step):

    def __init__(self, name, cwd, batch_settings):
        """Initialize a Cobalt qsub step

        :param name: name of the entity to launch
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param batch_settings: batch settings for entity
        :type batch_settings: BatchSettings
        """
        super().__init__(name, cwd)
        self.batch_settings = batch_settings
        self.step_cmds = []
        self.managed = True

    def get_launch_cmd(self):
        """Get the launch command for the batch

        :return: launch command for the batch
        :rtype: list[str]
        """
        script = self._write_script()
        return [self.batch_settings.batch_cmd, script]

    def add_to_batch(self, step):
        """Add a job step to this batch

        :param step: a job step instance e.g. SrunStep
        :type step: Step
        """
        launch_cmd = step.get_launch_cmd()
        self.step_cmds.append(launch_cmd)
        logger.debug(f"Added step command to batch for {step.name}")

    def _write_script(self):
        """Write the batch script

        :return: batch script path after writing
        :rtype: str
        """
        batch_script = self.get_step_file(ending=".sh")
        cobalt_debug = self.get_step_file(ending=".cobalt-debug")
        output, error = self.get_output_files()
        with open(batch_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#COBALT -o {output}\n")
            f.write(f"#COBALT -e {error}\n")
            f.write(f"#COBALT --cwd {self.cwd}\n")
            f.write(f"#COBALT --jobname {self.name}\n")
            f.write(f"#COBALT --debuglog {cobalt_debug}\n")

            # add additional sbatch options
            for opt in self.batch_settings.format_batch_args():
                f.write(f"#COBALT {opt}\n")

            for i, cmd in enumerate(self.step_cmds):
                f.write("\n")
                f.write(f"{' '.join((cmd))} &\n")
                if i == len(self.step_cmds)-1:
                    f.write("\n")
                    f.write("wait\n")
        os.chmod(batch_script, stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)
        return batch_script