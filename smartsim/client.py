from .connection import Connection
from .error import SmartSimError
import os

class Client:
    """The client class is used to communicate with various SmartSim entities that
       the user defines. Clients can hold multiple connection objects and therefore
       can send and recieve from multiple locations.

       SmartSim creates a distributed global dictionary for each Connection instance.
       The user can store and retrieve data from this global dictionary however they
       please.

       If multiple connections are registered to this specific client, each call to
       get_data(key) will retrieve all data that has been stored under that key for
       each connection.
    """
    def __init__(self):
        self.connections_out = []
        self.connections_in = []

    def get_data(self, key):
        all_data = []
        for conn in self.connections_in:
            data = conn.get(key)
            all_data.append(data)

        # if only one connection, dont return a list
        if len(all_data) == 1:
            all_data = all_data[0]
        return all_data

    def send_data(self, key, value):
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
                for db in data_out:
                    conn = Connection()
                    conn.connect(address, port, db)
                    self.connections_out.append(conn)
            except KeyError:
                raise SmartSimError("Orchestration", "No connections found for client!")
            try:
                # export SSDATAIN="1 2 3 4"
                data_in = [db for db in os.environ["SSDATAIN"].split()]
                for db in data_in:
                    conn = Connection()
                    conn.connect(address, port, db)
                    self.connections_in.append(conn)
            except KeyError:
                # TODO improve this error message
                raise SmartSimError("Orchestration", "No connections found for client!")
        except KeyError:
            raise SmartSimError("Orchestration", "No Orchestrator found in setup!")