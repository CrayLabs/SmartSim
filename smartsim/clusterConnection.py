
import time
from rediscluster import RedisCluster
from rediscluster.exceptions import RedisClusterException
from smartsim.error import SmartSimConnectionError


class ClusterConnection:
    """
        The ClusterConnection class is the same implementation as the Connection
        class save for the fact that the ClusterConnection class uses redis-py-cluster
        instead of redis-py for connection to clusters of KeyDB nodes.

        This class is used as the default connection for the Python client.
    """

    def __init__(self, db_id):
        self.conn = None
        self.db_id = db_id

    def connect(self, host, port):
        try:
            startup_nodes = [{"host": host, "port": port}]
            self.conn = RedisCluster(startup_nodes=startup_nodes)
            if not self.connected():
                raise SmartSimConnectionError(
                    "Could not reach orchestrator at " + host)
        except RedisClusterException as e:
            raise SmartSimConnectionError(
                "Could not reach orchestrator at " + host) from e

    def connected(self):
        """pings server and returns true if connected"""
        response = self.conn.ping()
        return response

    def get(self, key, wait=False, wait_interval=.5):
        # TODO put in a timeout limit
        prefixed_key = "_".join((self.db_id, key))
        data = self.conn.get(prefixed_key)
        if not data and wait:
            time.sleep(wait_interval)
            return self.get(key, wait=wait)
        else:
            return data

    def send(self, key, value):
        key = "_".join((self.db_id, key))
        self.conn.set(key, value)

