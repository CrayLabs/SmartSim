
import os

from ...utils import get_env, get_config
from ...error import SSConfigError
from itertools import product

class SlurmStep:

    def __init__(self, name, run_settings, multi_prog):
        """Initialize a job step for Slurm using Smartsim entity
           run_settings. Optionally make a multi-prog command by
           writing a multiple program configuration.

        :param name: identifer for the step instance
        :type name: str
        :param run_settings: smartsim run settings for an entity
        :type run_settings: dict
        :param multi_prog: make a multi-prog command, defaults to False
        :type multi_prog: bool, optional
        """
        self.name = name
        self.run_settings = run_settings
        self.multi_prog = multi_prog
        self._set_cwd()
        self.set_alloc()

    def set_alloc(self, alloc_id=None):
        """Read the run_settings of the entity provided by the
           controller and search for "alloc" keyword for setting
           the allocation

        :param alloc_id: id of the allocation, defaults to None
        :type alloc_id: str, optional
        :raises SSConfigError: if no allocation specified by the
                               user
        """
        if not alloc_id:
            self.alloc_id = get_config("alloc", self.run_settings, none_ok=True)
        else:
            self.alloc_id = alloc_id
        if not self.alloc_id:
            raise SSConfigError(f"No allocation specified for step")


    def build_cmd(self):
        """build the command to run the step with Slurm.

        :return: full srun command to run the job step
        :rtype: str
        """
        nodes = self.run_settings["nodes"]
        ppn = self.run_settings["ppn"]
        ntasks = ppn * self.run_settings["nodes"]
        out = self.run_settings["out_file"]
        err = self.run_settings["err_file"]

        srun = ["srun", "--nodes", str(nodes),
                        "--ntasks", str(ntasks),
                        "--ntasks-per-node", str(ppn),
                        "--output", out,
                        "--error", err,
                        "--jobid", str(self.alloc_id),
                        "--job-name", self.name]

        if "env_vars" in self.run_settings:
            env_var_str = self._format_env_vars(self.run_settings["env_vars"])
            srun += ["--export", env_var_str]

        smartsim_args = ["ppn", "nodes", "executable", "env_vars",
                         "exe_args", "out_file", "err_file", "cwd",
                         "alloc", "duration"]

        for opt, value in self.run_settings.items():
            if opt not in smartsim_args:
                prefix = "-" if len(str(opt)) == 1 else "--"
                if not value:
                    srun += [prefix + opt]
                else:
                    srun += ["=".join((prefix+opt, str(value)))]

        srun += self._build_exe_cmd()
        return srun


    def _build_exe_cmd(self):
        """Use smartsim arguments to construct the executable portion
           of the srun command.

        :raises SSConfigError: if executable argument is not provided
        :return: executable portion of srun command
        :rtype: str
        """
        try:
            exe = get_config("executable", self.run_settings)
            exe_args = get_config("exe_args", self.run_settings, none_ok=True)
            if self.multi_prog:
                cmd =  self._build_multi_prog_exe(exe, exe_args)
                return [" ".join(("--multi-prog", cmd))]
            else:
                if not exe_args:
                    exe_args = ""
                cmd = " ".join((exe, exe_args))

                return [cmd]

        except KeyError as e:
            raise SSConfigError(
                "SmartSim could not find following required field: %s" %
                (e.args[0])) from None


    def _build_multi_prog_exe(self, executable, exe_args):
        """Launch multiple programs on seperate CPUs on the same node using the
           slurm --multi-prog feature. Currently we only support launching one
           process with each execs and args. Execs and args are expected to be
           lists of the same length. Writes out a run_orc.conf.

           TODO: improve so that single exec or arg could be used
           TODO: catch bad formatting errors
           TODO: eliminate writing side effect of this function

           :type executable: list of strings
           :param exe_args: list of arguments to each executable
           :type exe_args: list of strings
           :returns: path to run_orc.conf
           :rtype: str
        """
        out = self.run_settings["out_file"]
        ppn = self.run_settings["ppn"]

        conf_path = os.path.join(os.path.dirname(out), "run_orc.conf")
        if not isinstance(executable, list):
            executable = [executable]
        if not isinstance(exe_args, list):
            exe_args = [exe_args]
        launch_args = list(product(executable, exe_args))
        with open(conf_path, "w+") as f:
            proc_num = 0
            for exe, arg in launch_args:
                f.write(" ".join((str(proc_num),  exe, arg, "\n")))
                proc_num += 1
        return conf_path

    def _format_env_vars(self, env_vars):
        """Slurm takes exports in comma seperated lists
           the list starts with all as to not disturb the rest of the environment
           for more information on this, see the slurm documentation for srun"""
        format_str = "".join(("PATH=", get_env("PATH"), ",",
                              "PYTHONPATH=", get_env("PYTHONPATH"), ","
                              "LD_LIBRARY_PATH=", get_env("LD_LIBRARY_PATH")))

        for k, v in env_vars.items():
            format_str += "," + "=".join((k,v))
        return format_str

    def _set_cwd(self):
        """return the cwd directory of a smartsim entity and set
           as the cwd directory of this step.

        :return: current working directory of the smartsim entity
        :rtype: str
        """
        self.cwd = self.run_settings["cwd"]
