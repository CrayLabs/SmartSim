from .connection import Connection
from .clusterConnection import ClusterConnection
from .error import SmartSimError, SSConfigError, SmartSimConnectionError
import pickle
import os

class Client:
    """The client class is used to communicate with various SmartSim entities that
       the user defines. Clients can hold multiple connection objects and therefore
       can send and recieve from multiple locations.

       If multiple connections are registered to this specific client, each call to
       get_data(key) will retrieve all data that has been stored under that key for
       each connection.
    """
    def __init__(self, cluster=True):
        self.connections_out = []
        self.connections_in = []
        self.cluster = cluster

    def get_data(self, key, wait=False, wait_interval=.5):
        """Get the value associated with some key from all the connections registered
           in the orchestrator

           Will return None if the key does not exist. Wait implies that the request
           should be made until a key by that name is stored in the connection.

           :param str key: key of the value being retrieved
           :param bool wait: flag for polling connection until value appears
           :param float wait_interval: seconds to wait between polling requests
           :returns bytes: bytes string of the data stored at key
           """
        all_data = []
        for conn in self.connections_in:
            data = conn.get(key, wait=wait, wait_interval=wait_interval)
            all_data.append(data)

        # if only one connection, dont return a list
        if len(all_data) == 1:
            all_data = all_data[0]
        return all_data

    def send_data(self, key, value):
        """Send bytes to be stored at some key.

        :param str key: string to store value at
        :param bytes value: bytes to store in orchestrator at key
        """
        if type(value) != bytes:
            raise SmartSimError("Value sent must be in bytes")
        if type(key) != str:
            raise SmartSimError("Key must be of string type")
        else:
            for conn in self.connections_out:
                conn.send(key, value)

    def setup_connections(self):
        """Retrieve the environment variables specific to this Client instance that have
           been registered by the user in the Orchestrator.
           Setup a connection for each of the registered Clients and leave all connections
           open for sending and recieving data
        """
        try:
            # export SSDB="127.0.0.1:6379"
            db_location = os.environ["SSDB"].split(":")
            address = db_location[0]
            port = db_location[1]
            try:
                # export SSDATAOUT="1 2 3 4"
                data_out = [db for db in os.environ["SSDATAOUT"].split()]
                for db_id in data_out:
                    if self.cluster:
                        conn = ClusterConnection(db_id)
                    else:
                        conn = Connection(db_id)
                    conn.connect(address, port)
                    self.connections_out.append(conn)
            except KeyError:
                raise SmartSimConnectionError("No connections found for client!")
            try:
                # export SSDATAIN="1 2 3 4"
                data_in = [db for db in os.environ["SSDATAIN"].split()]
                for db_id in data_in:
                    if self.cluster:
                        conn = ClusterConnection(db_id)
                    else:
                        conn = Connection(db_id)
                    conn.connect(address, port)
                    self.connections_in.append(conn)
            except KeyError:
                raise SmartSimConnectionError("No connections found for client!")
        except KeyError:
            raise SSConfigError("No Orchestrator found in setup!")