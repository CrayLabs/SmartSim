import os

from .step import Step


class LocalStep(Step):
    def __init__(self, name, cwd, run_settings):
        super().__init__(name, cwd)
        self.run_settings = run_settings
        self.env = self._set_env()

    def get_launch_cmd(self):
        cmd = []
        if self.run_settings.run_command:
            cmd.append(self.run_settings.run_command)
            run_args = self.run_settings.format_run_args()
            cmd.extend(run_args)

        cmd.extend(self.run_settings.exe)

        if self.run_settings.exe_args:
            cmd.extend(self.run_settings.exe_args)

        return cmd

    def _set_env(self):
        env = os.environ.copy()
        if self.run_settings.env_vars:
            for k, v in self.run_settings.env_vars.items():
                env[k] = v
        return env
