from subprocess import PIPE, Popen, CalledProcessError
from ..error.errors import LauncherError

from ..utils import get_logger
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

    def run(self, cmd, run_settings):
        """Launch a process using Popen

           :param str cmd: the command to run
           :param dict run_settings: a dictionary of settings for the subprocess
        """
        cmd = " ".join(cmd)
        out = run_settings["out_file"]
        err = run_settings["err_file"]
        cwd = run_settings["cwd"]
        process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, cwd=cwd)
        try:
            # waiting for the process to terminate and capture its output
            output, error = process.communicate()
            self.write_output(out, err, output, error)

        except CalledProcessError as e:
            raise LauncherError("Exception caught when running %s" %
                                (cmd)) from e

    def get_job_nodes(self, job_id):
        return ["127.0.0.1"]

    def stop(self, job_id):
        """Stop a process that is currently running"""
        raise NotImplementedError

    def get_sjob_stat(self, job_id):
        """Get the status of a job currenlty running"""
        raise NotImplementedError

    def write_output(self, out_file, err_file, output, error):
        """Write the output of a Popen subprocess"""
        with open(out_file, "wb+") as of:
            of.write(output)
        with open(err_file, "wb+") as ef:
            ef.write(error)
