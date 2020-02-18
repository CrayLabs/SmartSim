from .error import SSConfigError, SmartSimError
from .junction import Junction
from .dbnode import DBNode

from os import path, getcwd, environ


class Orchestrator:
    """The Orchestrator is responsible for launching the DataBase and connecting
       the various user specified entities(Clients, Models) to the correct
       endpoints
    """
    def __init__(self, orc_path, port=6379, cluster_size=3, partition=None):
        self.port = port
        self.path = orc_path
        self.junction = Junction()
        self.dbnodes = []
        self._init_db_nodes(cluster_size, partition)

    def _init_db_nodes(self, cluster_size, partition):
        cluster = True
        if cluster_size < 3:
            cluster = False
        run_settings = {"nodes": 1, "ppn": 1, "partition": partition}
        for node_id in range(cluster_size):
            node = DBNode(node_id,
                          self.path,
                          run_settings,
                          port=self.port,
                          cluster=cluster)
            self.dbnodes.append(node)

    def get_connection_env_vars(self, entity):
        """gets the environment variables from the junction for which databases each entity
           should connect to
        """
        return self.junction.get_connections(entity)

    def __str__(self):
        orc_str = "\n-- Orchestrator --"
        orc_str += str(self.junction)
        return orc_str

    def set_path(self, new_path):
        """Set the path for logging outputs for the database when user calls
           the generator.
        """
        self.path = new_path
        for dbnode in self.dbnodes:
            dbnode.set_path(new_path)
