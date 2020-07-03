
from ..shell import execute_cmd
from .localStep import LocalStep
from ...error.errors import LauncherError, SSUnsupportedError

from ...utils import get_logger
logger = get_logger(__name__)


class LocalLauncher:
    """Launcher used for spawning proceses on a localhost machine.
       Primiarly used for testing and prototying purposes, this launcher
       doesn't have the same capability as the launchers that inheirt from
       the SmartSim launcher base class as those launcher interact with the
       workload manager.

       All jobs will be launched serially and will not be able to be queried
       through the controller interface like jobs submitted to a workload
       manager like Slurm.
    """
    def __init__(self):
        pass

    def validate(self, nodes=None, ppn=None, partition=None):
        raise SSUnsupportedError("Local launcher does not support job validation")

    def create_step(self, name, run_settings, multi_prog=False):
        if multi_prog:
            raise SSUnsupportedError(
                "Local Launcher does not support mutliple program jobs")
        step = LocalStep(run_settings)
        return step

    def get_step_status(self, step_id):
        raise SSUnsupportedError("Local launcher does not support step statuses")

    def get_step_nodes(self, step_id):
        return ["127.0.0.1"]

    def accept_alloc(self, alloc_id):
        raise SSUnsupportedError("Local launcher does not support allocations")

    def free_alloc(self, alloc_id):
        raise SSUnsupportedError("Local launcher does not support allocations")

    def get_alloc(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        raise SSUnsupportedError("Local launcher does not support allocations")

    def run(self, step):
        """Run a local step created by this launcher. Utilize the shell
           library to execute the command with a Popen. Output and error
           files will be written to the entity path.

        :param step: LocalStep instance to run
        :type step: LocalStep
        """
        out_file = step.run_settings["out_file"]
        err_file = step.run_settings["err_file"]
        cmd = step.build_cmd()

        returncode, output, error = execute_cmd(cmd, shell=True,
                                                cwd=step.cwd, env=step.env)
        self._write_output(out_file, err_file, output, error)

    def stop(self, step_id):
        raise SSUnsupportedError("Local launcher does not support job interaction")

    def is_finished(self, status):
        raise SSUnsupportedError("Local launcher does not support job interaction")

    def _write_output(self, out_file, err_file, output, error):
        """Write the output of a Popen subprocess"""
        with open(out_file, "w+") as of:
            of.write(output)
        with open(err_file, "w+") as ef:
            ef.write(error)
