from .entity import SmartSimEntity
from .utils.helpers import get_env
import os


class DBNode(SmartSimEntity):
    def __init__(self, dbnodenum, path, run_settings, port=6379, cluster=True):
        name = "orchestrator_" + str(dbnodenum)
        super().__init__(name, path, run_settings)
        self.setup_dbnode(cluster, port)

    def setup_dbnode(self, cluster, port):
        conf_path = "".join((get_env("ORCCONFIG"), "smartsimdb.conf "))
        cluster_file = ""

        if cluster:
            conf_path += "--cluster-enabled yes "
            cluster_conf = "nodes-" + self.name + "-" + str(
                port) + ".conf"
            cluster_file = "--cluster-config-file " + os.path.join(
                self.path, cluster_conf)
        exe_args = " ".join((conf_path, "--port", str(port), cluster_file))

        new_settings = {
            "executable": "keydb-server",
            "exe_args": exe_args
        }
        self.update_run_settings(new_settings)