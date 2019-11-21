
from .error import SSConfigError, SmartSimError
from launcher import SlurmLauncher, PBSLauncher
from .junction import Junction

from os import path, getcwd, environ

class Orchestrator:
    """The Orchestrator is responsible for launching the DataBase and connecting
       the various user specified entities(Clients, Models) to the correct
       endpoints
    """

    def __init__(self, name=None, port=6379, **kwargs):
        self.name = "Orchestrator" # for the Controller
        self.port = port
        self.junction = Junction()
        self.settings = kwargs
        if name:
            self.name = name

    def get_launch_settings(self):
        conf_path = self._get_conf_path() + "smartsimdb.conf"
        exe_args = conf_path + " --port " + str(self.port)
        exe = "keydb-server"
        orc_path = getcwd()

        # run_command
        cmd = [" ".join((exe, exe_args))]

        self.settings["output_file"] = path.join(orc_path, "orchestrator.out")
        self.settings["err_file"] = path.join(orc_path, "orchestrator.err")
        self.settings["cmd"] = cmd

        return self.settings, orc_path

    def get_connection_env_vars(self, entity):
        """gets the environment variables from the junction for which databases each entity
           should connect to
        """
        return self.junction.get_connections(entity)

    def _get_conf_path(self):
        try:
            path = environ["ORCCONFIG"]
            return path
        except KeyError:
            raise SSConfigError(self.get_state(), "SmartSim environment not set up!")

    def __str__(self):
        orc_str = "\n-- Orchestrator --"
        orc_str += str(self.junction)
        return orc_str