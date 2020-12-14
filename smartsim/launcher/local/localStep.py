import os

from ...error import SSConfigError


class LocalStep:

    def __init__(self, run_settings):
        self.run_settings = run_settings
        self._set_cwd()
        self._set_env()

    def build_cmd(self):
        """Build a command for launch through a python subprocess.

        :raises SSConfigError: if run_command or executable are not
                               present in the entity run_settings
        :return: command for a subprocess
        :rtype: list of str
        """

        try:
            exe = self.run_settings.get("executable")
            exe_args = self.run_settings.get("exe_args", [])
            if exe_args:
                if isinstance(exe_args, str):
                    exe_args = exe_args.split()
                elif isinstance(exe_args, list):
                    correct_type = all([isinstance(arg, str) for arg in exe_args])
                    if not correct_type:
                        raise TypeError(
                            "Executable arguments given were not of type list or str"
                        )
            cmd = [exe] + exe_args
            return cmd

        except KeyError as e:
            raise SSConfigError(
                "SmartSim could not find following required field: %s" %
                (e.args[0])) from None

    def _set_cwd(self):
        """return the cwd directory of a smartsim entity and set
           as the cwd directory of this step.

        :return: current working directory of the smartsim entity
        :rtype: str
        """
        self.cwd = self.run_settings["cwd"]

    def _set_env(self):
        env = os.environ.copy()
        if "env_vars" in self.run_settings:
            env_vars = self.run_settings["env_vars"]
            for k, v in env_vars.items():
                env[k] = v
        self.env = env