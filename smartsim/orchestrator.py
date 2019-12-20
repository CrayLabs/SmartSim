
from .error import SSConfigError, SmartSimError
from .junction import Junction
from .entity import SmartSimEntity

from os import path, getcwd, environ

class Orchestrator(SmartSimEntity):
    """The Orchestrator is responsible for launching the DataBase and connecting
       the various user specified entities(Clients, Models) to the correct
       endpoints
    """

    def __init__(self, name=None, port=6379, run_settings=dict()):
        super().__init__("orchestrator", None, run_settings)
        self.port = port
        self.junction = Junction()

    def get_run_settings(self):
        conf_path = self._get_conf_path() + "smartsimdb.conf"
        exe_args = conf_path + " --port " + str(self.port)
        exe = "keydb-server"

        # run_command
        cmd = [" ".join((exe, exe_args))]

        self.run_settings["output_file"] = path.join(self.path, "orchestrator.out")
        self.run_settings["err_file"] = path.join(self.path, "orchestrator.err")
        self.run_settings["cmd"] = cmd

        return self.run_settings

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
            raise SSConfigError("SmartSim environment not set up!")

    def __str__(self):
        orc_str = "\n-- Orchestrator --"
        orc_str += str(self.junction)
        return orc_str