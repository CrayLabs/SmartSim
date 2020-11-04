from os import path, getcwd, environ

from ..entity import DBNode
from ..entity import EntityList
from ..error import SSConfigError, SmartSimError
from ..utils.helpers import get_env, expand_exe_path


class Orchestrator(EntityList):
    """The Orchestrator is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(self, orc_path, port, db_nodes=1, launcher="slurm", **kwargs):
        """Initialize an orchestrator for an Experiment.

        :param orc_path: path to output, error and conf files
        :type orc_path: str
        :param port: port of the orchestrator, defaults to 6379
        :type port: int, optional
        :param db_nodes: number of database instances to launch,
                         defaults to 1
        :type db_nodes: int, optional
        """
        self.port = int(port)
        self.path = orc_path
        self.launcher = launcher
        super().__init__("orchestrator", orc_path, db_nodes=db_nodes, **kwargs)

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator.

        The number of DBNode instances created depends on the value
        of the db_nodes argument passed to the orchestrator initialization.
        If multiple databases per node are requested through
        kwarg "dpn", the port is incremented by 1 starting at the
        port listed in the orchestrator initialization.

        #TODO update this docstring
        :raises SmartSimError: if invalid db_nodes is requested
        """
        dpn = kwargs.pop("dpn", 1)
        db_nodes = kwargs.pop("db_nodes", 1)
        cluster = False if db_nodes < 3 else True
        on_alloc = True if "alloc" in kwargs else False

        db_conf = self._find_db_conf()

        for db_id in range(db_nodes):
            # get correct run_settings for dbnode based on launcher type
            if self.launcher == "slurm":
                run_settings = self._build_slurm_run_settings(dpn, **kwargs)
            elif self.launcher == "local":
                run_settings = self._build_local_run_settings(**kwargs)

            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            ports = []
            exe_args = []
            for port_offset in range(dpn):
                next_port = int(self.port) + port_offset
                db_args = self._get_db_args(cluster, on_alloc, next_port, db_id)

                exe_args.append(" ".join((db_conf, "--port", str(next_port), db_args)))
                ports.append(next_port)

            # if only one database per node we only need one exe_args
            exe_args = exe_args[0] if dpn == 1 else exe_args
            run_settings["exe_args"] = exe_args

            name = "_".join((self.name, str(db_id)))
            node = DBNode(name, self.path, run_settings, ports=ports)

            self.entities.append(node)

    def set_path(self, new_path):
        """Set the path for logging db outputs when user calls the generator.

        :param new_path: The path to use for logging database outputs
        :type new_path: str
        """
        self.path = new_path
        for dbnode in self.entities:
            dbnode.set_path(new_path)

    def _build_slurm_run_settings(self, dpn, **kwargs):
        """Build run settings for the orchestrator when launching with
        the Slurm workload manager

        :param dpn: number of databases per node
        :type dpn: int
        :return: slurm run_settings for the orchestrator
        :rtype: dict
        """
        exe = self._find_db_exe()

        run_settings = {"executable": exe, "nodes": 1, "ntasks": dpn, **kwargs}
        return run_settings

    def _build_local_run_settings(self, **kwargs):
        """Build run settings for the orchestrator when launching with
        the local launcher
        """
        exe = self._find_db_exe()
        run_settings = {"executable": exe, **kwargs}
        return run_settings

    def remove_stale_files(self):
        """Remove old database files that may crash launch"""
        for dbnode in self.entities:
            dbnode.remove_stale_dbnode_files()

    def _find_db_exe(self):
        """Find the database executable for the orchestrator

        :raises SSConfigError: if env not setup for SmartSim
        :return: path to database exe
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        exe = path.join(sshome, "third-party/KeyDB/src/keydb-server")
        try:
            full_exe = expand_exe_path(exe)
            return full_exe
        except SSConfigError as e:
            msg = "Database not built/installed correctly. "
            msg += "Could not locate database executable"
            raise SSConfigError(msg) from None

    def _find_db_conf(self):
        """Find the database configuration file on the filesystem

        :raises SSConfigError: if env not setup for SmartSim
        :return: path to configuration file for the database
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        conf_path = path.join(sshome, "smartsim/database/smartsimdb.conf")
        if not path.isfile(conf_path):
            msg = "Could not locate database configuration file.\n"
            msg += f"looked at path {conf_path}"
            raise SSConfigError(msg)
        else:
            return conf_path

    def _get_db_args(self, cluster, on_alloc, port, db_id):
        """Create the arguments neccessary for cluster creation

        :param cluster: True if launching a cluster
        :type cluster: bool
        :param on_alloc: True if launching on allocation
        :type on_alloc: bool
        :param port: port of a dbnode instance
        :type port: int
        :param db_id: id of the dbnode instance
        :type db_id: int
        :return: redis db arguments for cluster
        :rtype: str
        """
        db_args = ""
        if cluster:
            cluster_conf = self._get_dbnode_conf_fname(port, db_id)
            db_args = " ".join(
                ("--cluster-enabled yes", "--cluster-config-file ", cluster_conf)
            )
        return db_args

    def _get_dbnode_conf_fname(self, port, db_id):
        """Returns the .conf file name for the given port number

        :param port: port of a dbnode instance
        :type port: int
        :param db_id: id of the dbnode instance
        :type db_id: int
        :return: the dbnode configuration file name
        :rtype: str
        """

        return "".join(("nodes-", self.name, "_", str(db_id), "-", str(port), ".conf"))
