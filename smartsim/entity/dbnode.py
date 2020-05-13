from .entity import SmartSimEntity
from ..utils.helpers import get_env
import os


class DBNode(SmartSimEntity):
    """DBNode objects are the entities that make up the orchestrator.
       Each database node can be launched in a cluster configuration
       and take launch multiple databases per node.

       To configure how each instance of the database operates, look
       into the smartsimdb.conf.
    """

    def __init__(self, dbnodenum, path, run_settings, port=6379, cluster=True):
        """Initialize a database node within an orchestrator.

        :param dbnodenum: database node id within orchestrator
        :type dbnodenum: int
        :param path: path to output and error files
        :type path: str
        :param run_settings: how dbnode should be run, set by orchestrator
        :type run_settings: dict
        :param port: starting port of the node, incremented for each
                     database per node, defaults to 6379
        :type port: int, optional
        :param cluster: toggle for cluster creation, defaults to True
        :type cluster: bool, optional
        """
        name = "orchestrator_" + str(dbnodenum)
        super().__init__(name, path, "db", run_settings)
        self.ports = []
        self.setup_dbnodes(cluster, port)

    def setup_dbnodes(self, cluster, port):
        """Initialize and construct the database instances for this
           specific database node. If ppn in run_settings is > 1,
           multiple instances of a database will be created on the
           same node.

           If not creating a cluster, an no allocation is provided
           by the user, assume this is a local launch and run the
           database node as a daemon so that one entity and the
           database can run simultaneously.

        :param cluster: toggle for cluster creation
        :type cluster: bool
        :param port: starting port of this node
        :type port: int
        """
        sshome = get_env("SMARTSIMHOME")
        conf_path = os.path.join(sshome, "smartsim/smartsimdb.conf")
        db_args = ""
        exe_args = []

        db_per_node = self.run_settings["ppn"]
        for db_id in range(db_per_node):
            next_port = port + db_id

            if cluster:
                db_args = self._create_cluster_args(next_port)
            else:
                # if we are launching on an allocation dont daemonize
                if not "alloc" in self.run_settings:
                    db_args = "--daemonize yes"

            exe_args.append(" ".join((conf_path, "--port", str(next_port), db_args)))
            self.ports.append(next_port)

        # if only one database to launch per node, were not launching
        # in multi-prog mode so take the only element of the args list
        if db_per_node == 1:
            exe_args = exe_args[0]

        new_settings = {
            "executable": os.path.join(sshome, "third-party/KeyDB/src/keydb-server"),
            "exe_args": exe_args
        }
        self.update_run_settings(new_settings)


    def _create_cluster_args(self, port):
        """Create the arguments neccessary for cluster creation

        :param port: port of a dbnode instance
        :type port: int
        :return: redis db arguments for cluster
        :rtype: str
        """
        cluster_conf = self._get_dbnode_conf_fname(port)
        db_args = " ".join(("--cluster-enabled yes",
                            "--cluster-config-file ", cluster_conf))
        return db_args

    def _get_dbnode_conf_fname(self, port):
        """Returns the .conf file name for the given port number

        :param port: port of a dbnode instance
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """

        return "nodes-" + self.name + "-" + str(port) + ".conf"

    def remove_stale_dbnode_files(self):
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            conf_file = "/".join((self.path, self._get_dbnode_conf_fname(port)))
            if os.path.exists(conf_file):
                os.remove(conf_file)

        if "out_file" in self.run_settings:
            out_file = self.run_settings["out_file"]
            if os.path.exists(out_file):
                os.remove(out_file)

        if "err_file" in self.run_settings:
            err_file = self.run_settings["err_file"]
            if os.path.exists(err_file):
                os.remove(err_file)
