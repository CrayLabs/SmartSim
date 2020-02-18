from .entity import SmartSimEntity
from .utils.helpers import get_env
import os


class DBNode(SmartSimEntity):
    def __init__(self, dbnodenum, path, run_settings, port=6379, cluster=True):
        name = "orchestrator_" + str(dbnodenum)
        super().__init__(name, path, run_settings)
        self.port = port
        self._cluster = cluster

    def get_run_settings(self):
        conf_path = "".join((get_env("ORCCONFIG"), "smartsimdb.conf "))
        cluster_file = ""

        if self._cluster:
            conf_path += "--cluster-enabled yes "
            cluster_conf = "nodes-" + self.name + "-" + str(
                self.port) + ".conf"
            cluster_file = "--cluster-config-file " + os.path.join(
                self.path, cluster_conf)
        else:
            conf_path += " --daemonize yes "

        exe_args = " ".join((conf_path, "--port", str(self.port)))
        exe = "keydb-server"
        cmd = [" ".join((exe, exe_args, cluster_file))]
        self.set_cmd(cmd)

        self.run_settings["out_file"] = os.path.join(self.path,
                                                     self.name + ".out")
        self.run_settings["err_file"] = os.path.join(self.path,
                                                     self.name + ".err")

        return self.run_settings
