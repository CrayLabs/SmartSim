from .error import SSConfigError, SmartSimError
from .junction import Junction
from .entity import DBNode

from os import path, getcwd, environ


class Orchestrator:
    """The Orchestrator is an in-memory database that can be launched
       alongside entities in SmartSim. Data can be transferred between
       entities by using one of the Python, C, C++ or Fortran clients
       within an entity.
    """
    def __init__(self, orc_path, port=6379, db_nodes=1, **kwargs):
        """Initialize an orchestrator for an Experiment.

        :param orc_path: path to output, error and conf files
        :type orc_path: str
        :param port: port of the orchestrator, defaults to 6379
        :type port: int, optional
        :param db_nodes: number of database instances to launch,
                         defaults to 1
        :type db_nodes: int, optional
        """
        self.port = port
        self.path = orc_path
        self.junction = Junction()
        self.dbnodes = []
        self._init_db_nodes(db_nodes, **kwargs)

    def _init_db_nodes(self, db_nodes, **kwargs):
        """Initialize DBNode instances for the orchestrator.

        The number of DBNode instances created depends on the value
        of the db_nodes argument passed to the orchestrator initialization.
        If multiple databases per node are requested through
        kwarg "dpn", the port is incremented by 1 starting at the
        port listed in the orchestrator initialization.

        :param db_nodes: number of DBNode instances to create
        :type db_nodes: int
        :raises SmartSimError: if invalid db_nodes is requested
        """
        if db_nodes == 2:
            raise SmartSimError(
                "Only clusters of size 1 and >= 3 are supported by Smartsim"
                )
        cluster = False if db_nodes < 3 else True
        # We need to remove dpn from kwargs because
        # it is not a valid command line argument.
        # "dpn" is set as "ppn" later.
        if "dpn" in kwargs:
            dpn = kwargs["dpn"]
            kwargs.pop("dpn")
        else:
            dpn = 1

        run_settings = {"nodes": 1, "ppn": dpn, **kwargs}
        for node_id in range(db_nodes):
            node = DBNode(node_id,
                          self.path,
                          run_settings,
                          port=self.port,
                          cluster=cluster)
            self.dbnodes.append(node)

    def get_connection_env_vars(self, entity):
        """Return the environment variables needed to launch an entity

        This function returns the environment variables needed to launch
        an entity with a connection to this orchestrator. The connections
        are registered and held within the junction class.

        :param entity: SmartSimEntity object to obtain connections for
        :type entity: SmartSimEntity
        :return: dictionary of environment variables
        :rtype: dict
        """
        return self.junction.get_connections(entity)

    def __str__(self):
        """Return user-readable string form of Orchestrator

        :return: user-readable string of the Orchestrator object
        :rtype: str
        """
        orc_str = "Name: Orchestrator \n"
        orc_str += "Number of Databases: " + str(len(self.dbnodes)) + "\n"
        return orc_str

    def set_path(self, new_path):
        """Set the path for logging db outputs when user calls the generator.

        :param new_path: The path to use for logging database outputs
        :type new_path: str
        """
        self.path = new_path
        for dbnode in self.dbnodes:
            dbnode.set_path(new_path)
