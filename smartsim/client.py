from .connection import Connection
from .error import SmartSimError
import pickle
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

    def peek(self, key):
        """Check if the data has made it to the orchestrator yet

           :param str key: key of the value being retrieved
           :returns bool: True if data exists within orchestrator for all connections
        """
        for conn in self.connections_in:
            data = conn.get(key)
            if not data: # if get returns None
                return False
        return True

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
            if self._is_multipart(data):
                multipart_data = self._get_multipart(key, data, conn)
                all_data.append(multipart_data)
            else:
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

    def send_big_data(self, key, value, num_split=4):
        """Send a big piece of data over the network in many small byte packets.
           The orchestrator currently uses 4 threads so a num_split of 4 is optimal.

           :param str key: key to store data at for retrieval
           :param bytes value: bytes to store
           :param int num_split: number of splits in byte string
        """
        if type(value) != bytes:
            raise SmartSimError("Value sent must be in bytes")
        else:
            # send header to be stored
            multipart_header = pickle.dumps("multipart:" + str(num_split))
            self.send_data(key, multipart_header)

            # int divide bytes string
            bytes_per_msg = int(len(value)/num_split)

            # send num_splits # of messages to orchestrator
            # split by sliding a window over the bytes msg
            start_window = 0
            end_window = bytes_per_msg
            last_split = num_split - 1
            for split in range(num_split):
                split_key = ":".join((key, str(split)))
                if split == last_split:
                    self.send_data(split_key, value[start_window:])
                else:
                    self.send_data(split_key, value[start_window:end_window])
                start_window += bytes_per_msg
                end_window += bytes_per_msg

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
                raise SmartSimError("No connections found for client!")
            try:
                # export SSDATAIN="1 2 3 4"
                data_in = [db for db in os.environ["SSDATAIN"].split()]
                for db in data_in:
                    conn = Connection()
                    conn.connect(address, port, db)
                    self.connections_in.append(conn)
            except KeyError:
                # TODO improve this error message
                raise SmartSimError("No connections found for client!")
        except KeyError:
            raise SmartSimError("No Orchestrator found in setup!")


    def _is_multipart(self, data):
        # if its clearly not a multipart header dont
        # waste time unpickling it
        if data: # if not None
            if len(data) < 30:
                try:
                    header = pickle.loads(data)
                    if header.startswith("multipart:"):
                        return True
                # pickle throws keyerror usually when its not an object
                # that was pickled, so its not a header for a multipart
                except KeyError:
                    return False
            else:
                return False
        return False

    def _get_multipart(self, key, data, connection):
        """Get the various splits of a multipart message from the orchestrator"""
        # figure out the number of splits and split
        # get the data, concat together and send back.
        multipart_data = b""
        header = pickle.loads(data)
        num_splits = int(header.split(":")[1])

        for split in range(num_splits):
            split_key = ":".join((key, str(split)))
            split_data = connection.get(split_key, wait=True)
            multipart_data += split_data
        return multipart_data