
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

    def __init__(self):
        """Initialize a ClusterConnection
        """
        self.conn = None

    def connect(self, host, port):
        """Establish RedisCluster connection

        :param host: The host id to connect to
        :type host: str
        :param port: The port to connect to
        :type port: int
        :raises SmartSimConnectionError: if connection can't be established
        """
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
        """Pings server and returns true if connected

        :returns: True if server can be reached, otherwise false
        :rtype: bool
        """
        response = self.conn.ping()
        return response

    def get(self, key, wait=False, wait_interval=.5):
        """Retrieve a value from the database

        :param key: The key to retrieve from the database
        :type key: str
        :param wait: wait for the key to exists before returning
        :type wait: bool
        :param wait_interval: the frequency of checks for the key
        :type wait_interval: float
        :returns: data corresponding to the key
        :rytpe: serialized data
        """
        data = self.conn.get(key)
        if not data and wait:
            time.sleep(wait_interval)
            return self.get(key, wait=wait)
        else:
            return data

    def send(self, key, value):
        """Send data to the database

        :param key: The key associated with the value to send
        :type key: str
        :param value: The value to send
        :type value: serialized data
        """
        self.conn.set(key, value)

    def exists(self, key):
        """Check if a key exists in the database

        :param key: The key to check for
        :type key: str
        :returns: True if the key exists, otherwise false
        :rtype: boolean
        """
        return self.conn.exists(key)
