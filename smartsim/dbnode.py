from .entity import SmartSimEntity
from .utils.helpers import get_env
import os


class DBNode(SmartSimEntity):
    def __init__(self, dbnodenum, path, run_settings, port=6379, cluster=True):
        name = "orchestrator_" + str(dbnodenum)
        super().__init__(name, path, "db", run_settings)
        self.ports = []
        self.setup_dbnode(cluster, port)

    def setup_dbnode(self, cluster, port):
        sshome = get_env("SMARTSIMHOME")
        conf_path = os.path.join(sshome, "smartsim/smartsimdb.conf")
        cluster_file = ""
        exe_args = []

        dpn = self.run_settings["ppn"]
        for db in range(dpn):
            next_port = port + db

            if cluster:
                conf_path += " --cluster-enabled yes"
                cluster_conf = "nodes-" + self.name + "-" + str(next_port) + ".conf"
                cluster_file = " ".join(("--cluster-config-file ", cluster_conf))
            exe_args.append(" ".join((conf_path, "--port", str(next_port), cluster_file)))
            self.ports.append(next_port)

        if dpn == 1:
            exe_args = exe_args[0]

        new_settings = {
            "executable": os.path.join(sshome, "third-party/KeyDB/src/keydb-server"),
            "exe_args": exe_args
        }
        self.update_run_settings(new_settings)
