import os
from itertools import product

from ...error import LauncherError, SSConfigError
from ...utils import get_env
from ...utils.helpers import expand_exe_path


class SlurmStep:
    def __init__(self, name, run_settings, multi_prog):
        """Initialize a job step for Slurm

        This function initializes a job step using Smartsim entity
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
        self._set_alloc()

    def build_cmd(self):
        """Build the command to run the step with Slurm.

        :return: full srun command to run the job step
        :rtype: str
        """
        out = self.run_settings["out_file"]
        err = self.run_settings["err_file"]
        srun = self._find_srun_cmd()

        # start srun command creation.
        # add job information and output files
        step = [
            srun,
            "--output",
            out,
            "--error",
            err,
            "--jobid",
            str(self.alloc_id),
            "--job-name",
            self.name,
        ]

        if "env_vars" in self.run_settings:
            env_var_str = self._format_env_vars(self.run_settings["env_vars"])
            step += ["--export", env_var_str]

        # some kept here for deprecation sake
        smartsim_args = [
            "ppn",
            "executable",
            "env_vars",
            "exe_args",
            "out_file",
            "err_file",
            "cwd",
            "alloc",
            "duration",
        ]

        # add additional slurm arguments based on key length
        for opt, value in self.run_settings.items():
            if opt not in smartsim_args:
                short_arg = bool(len(str(opt)) == 1)
                prefix = "-" if short_arg else "--"
                if not value:
                    step += [prefix + opt]
                else:
                    if short_arg:
                        step += [prefix + opt, str(value)]
                    else:
                        step += ["=".join((prefix + opt, str(value)))]

        step += self._build_exe_cmd()
        return step

    def _build_exe_cmd(self):
        """Use smartsim arguments to construct executable portion of srun command.

        :raises SSConfigError: if executable argument is not provided
        :return: executable portion of srun command
        :rtype: str
        """
        try:
            exe = self.run_settings["executable"]
            exe_args = self.run_settings.get("exe_args", None)
            if self.multi_prog:
                cmd = self._build_multi_prog_exe(exe, exe_args)
                mp_cmd = ["--multi-prog", cmd]
                return mp_cmd
            if exe_args:
                if isinstance(exe_args, str):
                    exe_args = exe_args.split()
                elif isinstance(exe_args, list):
                    correct_type = all([isinstance(arg, str) for arg in exe_args])
                    if not correct_type:
                        raise TypeError(
                            "Executable arguments given were not of type list or str"
                        )
                else:
                    exe_args = [""]
                cmd = [exe] + exe_args
                return cmd
            return [exe]

        except KeyError as e:
            raise SSConfigError(
                f"SmartSim could not find following required field: {e.args[0]}"
            ) from None

    def _build_multi_prog_exe(self, executable, exe_args):
        """Build Slurm multi prog executable

        Launch multiple programs on seperate CPUs on the same node using the
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

        conf_path = os.path.join(os.path.dirname(out), "run_orc.conf")
        if not isinstance(executable, list):
            executable = [executable]
        if not isinstance(exe_args, list):
            exe_args = [exe_args]
        launch_args = list(product(executable, exe_args))
        with open(conf_path, "w+") as f:
            proc_num = 0
            for exe, arg in launch_args:
                f.write(" ".join((str(proc_num), exe, arg, "\n")))
                proc_num += 1
        return conf_path

    def _find_srun_cmd(self):
        try:
            srun = expand_exe_path("srun")
            return srun
        except SSConfigError:
            raise LauncherError(f"Slurm launcher could not find srun executable path") from None

    def _format_env_vars(self, env_vars):
        """Build environment variable string for Slurm

        Slurm takes exports in comma seperated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun
        :param env_vars: additional environment variables to add
        :type env_vars: list of str, optional
        :returns: the formatted string of environment variables
        :rtype: str
        """
        format_str = "".join(
            (
                "PATH=",
                get_env("PATH"),
                ",",
                "PYTHONPATH=",
                get_env("PYTHONPATH"),
                ",",
                "LD_LIBRARY_PATH=",
                get_env("LD_LIBRARY_PATH"),
            )
        )

        for k, v in env_vars.items():
            format_str += "," + "=".join((k, str(v)))
        return format_str

    def _set_cwd(self):
        """Set the current working directory of the step"""
        self.cwd = self.run_settings["cwd"]

    def _set_alloc(self):
        """Set the allocation id of the job step

        Read the run_settings of the entity provided by the
        controller and search for "alloc" keyword for setting
        the allocation

        :raises SSConfigError: if no allocation specified by the user
        """
        alloc = self.run_settings.get("alloc", None)
        if alloc:
            self.alloc_id = str(alloc)
        else:
            raise SSConfigError(f"No allocation specified for step")
