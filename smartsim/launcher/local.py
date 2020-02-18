import asyncio
from asyncio.subprocess import PIPE


class LocalLauncher:
    """Launcher used for spawning proceses on a localhost machine.
       Primiarly used for testing and prototying purposes, this launcher
       doesn't have the same capability as the launchers that inheirt from
       the SmartSim launcher base class as those launcher interact with the
       workload manager.
    """
    def __init__(self):
        self.processes = dict()

    def run(self, cmd, run_settings):
        """Launch a process using asyncio

           :param str cmd: the command to run
           :param dict run_settings: a dictionary of settings for the subprocess
        """
        cmd = " ".join(cmd)
        out = run_settings["out_file"]
        err = run_settings["err_file"]
        cwd = run_settings["cwd"]
        pid = asyncio.run(async_run(cmd, out, err, cwd))
        return pid

    def get_job_nodes(self, job_id):
        return ["127.0.0.1"]

    def stop(self, job_id):
        """Stop a process that is currently running"""
        raise NotImplementedError

    def get_sjob_stat(self, job_id):
        """Get the status of a job currenlty running"""
        raise NotImplementedError


async def async_run(cmd, out, err, cwd):
    proc = await asyncio.create_subprocess_shell(cmd,
                                                 cwd=cwd,
                                                 stderr=PIPE,
                                                 stdout=PIPE)
    stdout, stderr = await proc.communicate()
    if stdout:
        with open(out, mode="wb") as f:
            f.write(stdout)
    if stderr:
        with open(err, mode="wb") as g:
            g.write(stdout)
    return proc.pid
