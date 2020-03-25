from ..connection import Connection
from ..clusterConnection import ClusterConnection
from ..error import SmartSimError, SSConfigError, SmartSimConnectionError
from ..utils.protobuf.smartsim_protobuf_pb2 import ArrayDouble, ScalarDouble, ScalarSInt64
import pickle
import os
import numpy as np
import time

class Client:
    """The client class is used to communicate with various SmartSim entities that
       the user defines. Clients can hold multiple connection objects and therefore
       can send and recieve from multiple locations.

       If multiple connections are registered to this specific client, each call to
       get_data(key) will retrieve all data that has been stored under that key for
       each connection.
    """
    def __init__(self, cluster=True):
        self.connections_in = dict()
        self.connection_out = None
        self.cluster = cluster
        self.name = None

    def get_array_nd_float64(self, key, wait=False, wait_interval=1000):
        """"This function gets a numpy array of type float64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               numpy.ndarray or single numpy.ndarray
        """
        return self._get_data(key, ArrayDouble(), wait, wait_interval)

    def get_array_nd_int64(self, key, wait=False, wait_interval=1000):
        """"This function gets a numpy array of type int64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               numpy.ndarray or single numpy.ndarray
        """
        return self._get_data(key, ArraySInt64(), wait, wait_interval)

    def get_scalar_float64(self, key, wait=False, wait_interval=1000):
        """"This function gets a scalar of type float64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               float64 or single float64
        """
        return self._get_data(key, ScalarDouble(), wait, wait_interval)

    def get_scalar_int64(self, key, wait=False, wait_interval=1000):
        """"This function gets a scalar of type int64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               int64 or single int64
        """
        return self._get_data(key, ScalarSInt64(), wait, wait_interval)

    def put_array_nd_float64(self, key, value):
        """This function puts a numpy array of type float64 into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: numpy.ndarray filled with float64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_float64() must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('float64')):
            raise SmartSimError("The values inside of the numpy.ndarray must be of type float64 to use put_array_nd_float64().")

        self._put_data(key, value, ArrayDouble())

    def put_array_nd_int64(self, key, value):
        """This function puts a numpy array of type int64 into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: numpy.ndarray filled with int64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int64() must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int64')):
            raise SmartSimError("The values inside of the numpy.ndarray must be of type int64 to use put_array_nd_int64().")

        self._put_data(key, value, ArraySInt64())

    def put_scalar_float64(self, key, value):
        """This function puts a 64 bit float into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: float
        """

        if not isinstance(value, float):
            raise SmartSimError("The value passed into put_scalar_float64() must be of type float")

        self._put_data(key, value, ScalarDouble())

    def put_scalar_int64(self, key, value):
        """This function puts a 64 bit integer into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: int
        """

        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int64() must be of type int")

        self._put_data(key, value, ScalarSInt64())

    def exists(self, key):
        """This function checks if a key is in the database through any of the
           incoming connections
           :param str key: key to check in the database
           :returns key_exists: true if the key is in the database, false otherwise
           :rtype key_exists: bool
        """
        key_exists = False
        for name, conn in self.connections_in.items():
            prefixed_key = "_".join((name, key))
            if conn.exists(prefixed_key):
                key_exists = True
        return key_exists

    def poll_key(self, key, poll_frequency=1000, num_tries=-1):
        """This function polls for a key with a specificed frequency
           and specified number of times
           :param str key: key to check for in the database
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tires: the maximum number of tries.  If -1, unlimited
                                 attempts will be made.
        """
        key_exists = False
        while (not key_exists) and (not num_tries==0):
            for name, conn in self.connections_in.items():
                prefixed_key = "_".join((name, key))
                if conn.exists(prefixed_key):
                    key_exists = True
                else:
                    if(num_tries>0):
                        num_tries-=1
                    time.sleep(poll_frequency/1000.0)

        return key_exists

    def poll_key_and_check_scalar_float64(self, key, value, poll_frequency=1000, num_tries=-1):
        """Poll key to check existence and if it is exists check against value
           :param str key: key to check for in the database
           :param flaot value: value to compare to
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tries: the maximum number of tries.  If -1, unlimited
                                 attempts will be made.
        """
        matched_value = False
        current_value = None

        while not num_tries == 0:

            for name, conn in self.connections_in.items():
                prefixed_key = "_".join((name, key))
                if conn.exists(prefixed_key):
                    current_value = self.get_scalar_float64(prefixed_key)
                    if current_value == value:
                        num_tries = 0
                        matched_value = True

            if not matched_value:
                time.sleep(poll_frequency/1000.0)
            if num_tries>0:
                num_tries -= 1

        return matched_value

    def poll_key_and_check_scalar_int64(self, key, value, poll_frequency=1000, num_tries=-1):
        """Poll key to check existence and if it is exists check against value
           :param str key: key to check for in the database
           :param int value: value to compare to
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tries: the maximum number of tries.  If -1, unlimited
                                 attempts will be made.
        """
        matched_value = False
        current_value = None

        while not num_tries == 0:

            for name, conn in self.connections_in.items():
                prefixed_key = "_".join((name, key))
                if conn.exists(prefixed_key):
                    current_value = self.get_scalar_int64(key)
                    if current_value == value:
                        num_tries = 0
                        matched_value = True

            if not matched_value:
                time.sleep(poll_frequency/1000.0)

            if num_tries>0:
                num_tries -= 1

        return matched_value

    def _get_data(self, key,  pb_message, wait=False, wait_interval=1000):
        """Get the value associated with some key from all the connections registered
           in the orchestrator

           Will return empty dictionary if the key does not exist. Wait implies that
           the request should be made until a key by that name is stored in the connection.

           :param str key: key of the value being retrieved
           :param str dtype: the numpy data type of the serialized data (e.g. float64)
           :param bool wait: flag for polling connection until value appears
           :param float wait_interval: milliseconds to wait between polling requests
           :returns all_data: dictionary of key and values or single value if only
                              one connection produces a result
           """
        if wait:
            self.poll_key(key, poll_frequency=wait_interval, num_tries=-1)

        all_data = {}
        for name, conn in self.connections_in.items():
            prefixed_key = "_".join((name, key))
            data_bytes = conn.get(prefixed_key, wait=wait, wait_interval=wait_interval)
            data = self.deserialize(pb_message, data_bytes)
            all_data[name] = data
        if len(all_data.keys()) == 1:
            return list(all_data.values())[0]
        else:
            return all_data

    def _put_data(self, key, value, pb_message):
        """This function puts a value into the database.
           It is an internal function not meant for user.
           :param str key: key of the value being put
           :param value: value being put
           :type value: int, float, or numpy.ndarray
           :param pb_message: protobuf message object
           :type pb_message: google.protobuf.pb_message
        """

        if type(key) != str:
            raise SmartSimError("Key must be of string type")
        if not self.connection_out:
            raise SmartSimError("Connection to database was not setup at runtime")

        value_bytes = self.serialize(pb_message, value)
        prefixed_key = "_".join((self.name, key))
        self.connection_out.send(prefixed_key, value_bytes)

    def serialize(self, pb_message, value):
        """Serializes value using protocol buffer
            :param pb_message: protobuf message object
            :type pb_message: google.protobuf.message
            :param value: data to serialize
            :type value: numpy.ndarray or scalar
            TODO let user specify row vs column major
        """

        if isinstance(value, np.ndarray):
            dimensions = list(value.shape)
            for dim in dimensions:
                pb_message.dimension.append(dim)
                
            for d in value.ravel():
                pb_message.data.append(d)
        else:
            pb_message.data = value

        return pb_message.SerializeToString()

    def deserialize(self, pb_message, data_bytes):
        """Unpacks the serialized protocol buffer into a numpy.ndarray
            :param pb_message: protobuf message objec to decode bytes
            :type pb_message: google.protobuf.message
            :param data_byes: the serialized data
            :type value: string
            :returns: numpy.ndarray, float, or int of the protocol buffer
            :rtype: numpy.ndarray, float, or int
        """

        pb_message.ParseFromString(data_bytes)

        if hasattr(pb_message, 'dimension'):
            dimensions = pb_message.dimension
            data = np.reshape(pb_message.data, dimensions)
        else:
            data = pb_message.data

        return data

    def setup_connections(self):
        """Retrieve the environment variables specific to this Client instance that have
           been registered by the user in the Orchestrator.
           Setup a connection for each of the registered Clients and leave all connections
           open for sending and recieving data
        """
        try:
            # export SSNAME="sim_one"
            self.name = os.environ["SSNAME"]
            # export SSDB="127.0.0.1:6379"
            db_location = os.environ["SSDB"].split(":")
            address = db_location[0]
            port = db_location[1]

            # setup data out connection to orchestrator
            if self.cluster:
                self.connection_out = ClusterConnection()
            else:
                self.connection_out = Connection()
            self.connection_out.connect(address, port)

            try:
                # export SSDATAIN="sim_one:sim_two"
                data_in = [connect for connect in os.environ["SSDATAIN"].split(":")]
                for connection in data_in:
                    if self.cluster:
                        conn = ClusterConnection()
                    else:
                        conn = Connection()
                    conn.connect(address, port)
                    self.connections_in[connection] = conn
            except KeyError:
                raise SmartSimConnectionError("No connections found for client!")
        except KeyError:
            raise SSConfigError("No Orchestrator found in setup!")

    def get_protobuf_message(self, value, dtype):
        """This function returns the correct protobuf message object
           based on the type of value.  Note that python floats
           are 64 bit and automatically assigned to double type.
            :param value: the value to be checked
            :type value: numpy.ndarray, float, double
            :param dtype: string of the data type expected in protocol buffer
            :type dtype: string
            :returns: protobuf message object
            :rtype: protobuf.
        """

        pb_value = None
        if isinstance(value, np.ndarray):
            if dtype.lower() == "float64":
                pb_value = ArrayDouble()
            elif dtype.lower() == "float32":
                pb_value = ArrayFloat()
            elif dtype.lower() == "int64":
                pb_value = ArrayInt64()
            elif dtype.lower() == "int32":
                pb_value = ArrayInt32
            elif dtype.lower() == "uint64":
                pb_value = ArrayUInt64
            elif dtype.lower() == "uint32":
                pb_value = ArrayUInt32
            else:
                raise SmartSimError("Numpy data type " + dtype + " not supported by protobuf messaging implementation in SmartSim.")
        else:
            if dtype.lower() == "float":
                pb_value = ScalarDouble
            elif dtype.lower() == "int":
                pb_value = ScalarSInt64
            else:
                raise SmartSimError("Scalar data type " + dtype + " not supported by the protobuf implementation in SmartSim.")
                
        return pb_value
