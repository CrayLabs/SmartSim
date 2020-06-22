from ..connection import Connection
from ..clusterConnection import ClusterConnection
from ..error import SmartSimError, SSConfigError, SmartSimConnectionError
from ..utils.protobuf.smartsim_protobuf_pb2 import ArrayDouble, ScalarDouble, ScalarSInt64, ScalarSInt32, ArraySInt32, ArraySInt64
import pickle
import os
import numpy as np
import time

class Client:
    """The client class is used to communicate with various SmartSim entities that
       the user defines. Clients can hold multiple connection objects and therefore
       can send and receive from multiple locations.

       If multiple connections are registered to this specific client, each call to
       get_data(key) will retrieve all data that has been stored under that key for
       each connection.
    """
    def __init__(self, cluster=True):
        self.cluster = cluster
        self.put_key_prefix = None
        self.get_key_prefix = None
        self.get_key_prefixes = []
        self.db_connection = None

    def get_array_nd_float64(self, key, wait=False, wait_interval=1000):
        """"This function gets a numpy array of type float64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               numpy.ndarray or single numpy.ndarray
        """
        return self._get_data(key, ArrayDouble(), True, wait, wait_interval)

    def get_array_nd_int64(self, key, wait=False, wait_interval=1000):
        """"This function gets a numpy array of type int64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               numpy.ndarray or single numpy.ndarray
        """
        return self._get_data(key, ArraySInt64(), True, wait, wait_interval)

    def get_array_nd_int32(self, key, wait=False, wait_interval=1000):
        """"This function gets a numpy array of type int32 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               numpy.ndarray or single numpy.ndarray
        """
        return self._get_data(key, ArraySInt32(), True, wait, wait_interval)

    def get_scalar_float64(self, key, wait=False, wait_interval=1000):
        """"This function gets a scalar of type float64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :    param float wait_interval: milliseconds to wait between polling requests
            :    returns all_data: dictionary of key connections_in.key() and value
                               float64 or single float64
        """
        return self._get_data(key, ScalarDouble(), True, wait, wait_interval)

    def get_scalar_int64(self, key, wait=False, wait_interval=1000):
        """"This function gets a scalar of type int64 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               int64 or single int64
        """
        return self._get_data(key, ScalarSInt64(), True, wait, wait_interval)

    def get_scalar_int32(self, key, wait=False, wait_interval=1000):
        """"This function gets a scalar of type int32 from the database.
            :param str key: key of the value to fetch
            :param bool wait: flag for polling connection until value appears
            :param float wait_interval: milliseconds to wait between polling requests
            :returns all_data: dictionary of key connections_in.key() and value
                               int32 or single int32
        """
        return self._get_data(key, ScalarSInt32(), True, wait, wait_interval)

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

        self._put_data(key, value, ArrayDouble(), True)

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

        self._put_data(key, value, ArraySInt64(), True)

    def put_array_nd_int32(self, key, value):
        """This function puts a numpy array of type int32 into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: numpy.ndarray filled with int32
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int32() must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int32')):
            raise SmartSimError("The values inside of the numpy.ndarray must be of type int32 to use put_array_nd_int32().")

        self._put_data(key, value, ArraySInt64(), True)

    def put_scalar_float64(self, key, value):
        """This function puts a 64 bit float into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: float
        """

        if not isinstance(value, float):
            raise SmartSimError("The value passed into put_scalar_float64() must be of type float")

        self._put_data(key, value, ScalarDouble(), True)

    def put_scalar_int64(self, key, value):
        """This function puts a 64 bit integer into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: int
        """

        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int64() must be of type int")

        self._put_data(key, value, ScalarSInt64(), True)

    def put_scalar_int32(self, key, value):
        """This function puts a 32 bit integer into the database.
           :param str key: key of the value being put
           :param value: value being put
           :type value: int
        """

        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int32() must be of type int")

        self._put_data(key, value, ScalarSInt32(), True)

    def exact_key_exists(self, key):
        """This function checks if a key is in the database
           :param str key: key to check in the database
           :returns key_exists: true if the key is in the database, false otherwise
           :rtype key_exists: bool
        """
        return self.db_connection.exists(key)

    def key_exists(self, key):
        """This function checks if a key is in the database through any of the
           incoming connections
           :param str key: key to check in the database
           :returns key_exists: true if the key is in the database, false otherwise
           :rtype key_exists: bool
        """
        return self.db_connection.exists(self._build_get_key(key))

    def poll_exact_key(self, key, poll_frequency=1000, num_tries=-1):
        """This function polls for a key without prefixing at a specificed frequency
           and specified number of times
           :param str key: key to check for in the database
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tires: the maximum number of tries.  If -1, unlimited
                                 attempts will be made.
        """
        key_exists_flag = False
        while (not key_exists_flag) and (not num_tries==0):
            if self.exact_key_exists( key ):
               return True
            else:
                if(num_tries>0):
                    num_tries-=1
                time.sleep(poll_frequency/1000.0)

        return False

    def poll_key(self, key, poll_frequency=1000, num_tries=-1):
        """This function polls for a key with a specificed frequency
           and specified number of times
           :param str key: key to check for in the database
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tires: the maximum number of tries.  If -1, unlimited
                                 attempts will be made.
        """
        key_exists_flag = False
        while (not key_exists_flag) and (not num_tries==0):
            if self.key_exists( key ):
               return True
            else:
                if(num_tries>0):
                    num_tries-=1
                time.sleep(poll_frequency/1000.0)

        return False

    def poll_key_and_check_scalar_float64(self, key, value, poll_frequency=1000, num_tries=-1):
        """Poll key to check existence and if it is exists check against value
           :param str key: key to check for in the database
           :param float value: value to compare to
           :param float poll_frequency: the time in milliseconds beteween tries
           :param int num_tries: the maximum number of tries.  If <0, unlimited
                                 attempts will be made.
        """
        matched_value = False
        current_value = None

        while not num_tries == 0:
            if self.key_exists(key):
                current_value = self.get_scalar_float64(key)
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
            if self.key_exists(key):
                current_value = self.get_scalar_int64(key)
                if current_value == value:
                    num_tries = 0
                    matched_value = True

            if not matched_value:
                time.sleep(poll_frequency/1000.0)

            if num_tries>0:
                num_tries -= 1

        return matched_value

    def poll_key_and_check_scalar_int32(self, key, value, poll_frequency=1000, num_tries=-1):
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
            if self.key_exists(key):
                current_value = self.get_scalar_int32(key)
                if current_value == value:
                    num_tries = 0
                    matched_value = True

            if not matched_value:
                time.sleep(poll_frequency/1000.0)

            if num_tries>0:
                num_tries -= 1

        return matched_value

    def set_data_source( self, prefix ):
        """Set the prefix used for key queries from the database. The prefix is
           checked to ensure that it is a valid data source
        """
        if prefix in self.get_key_prefixes:
            self.get_key_prefix = prefix
        else:
            raise SmartSimError(f"{prefix} has not been registered")

    def _get_data(self, key, pb_message, add_prefix, wait=False, wait_interval=1000):
        """Get the value associated with some key from all the connections registered
           in the orchestrator

           Will return empty dictionary if the key does not exist. Wait implies that
           the request should be made until a key by that name is stored in the connection.

           :param str key: key of the value being retrieved
           :param str dtype: the numpy data type of the serialized data (e.g. float64)
           :param logical add_prefix: if true, add a prefix to the key
           :param bool wait: flag for polling connection until value appears
           :param float wait_interval: milliseconds to wait between polling requests
           :returns data: dictionary of key and values or single value if only
                              one connection produces a result
           """

        if add_prefix:
            query_key = self._build_get_key(key)
        else:
            query_key = key

        if wait:
            self.poll_exact_key(query_key, poll_frequency=wait_interval, num_tries=-1)


        data_bytes = self.db_connection.get(query_key, wait=wait, wait_interval=wait_interval)
        data = self.deserialize(pb_message, data_bytes)

        return data

    def _put_data(self, key, value, pb_message, add_prefix):
        """This function puts a value into the database.
           It is an internal function not meant for user.
           :param str key: key of the value being put
           :param value: value being put
           :type value: int, float, or numpy.ndarray
           :param logical add_prefix: if true, add a prefix to the key
           :param pb_message: protobuf message object
           :type pb_message: google.protobuf.pb_message
           :param logical add_prefix: if true, potentially add a prefix to the key
        """

        if type(key) != str:
            raise SmartSimError("Key must be of string type")
        if not self.db_connection:
            raise SmartSimError("Connection to database was not setup at runtime")

        value_bytes = self.serialize(pb_message, value)
        send_key = key
        if self.put_key_prefix and add_prefix:
            send_key = "_".join((self.put_key_prefix, key))
        self.db_connection.send(send_key, value_bytes)

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
           open for sending and receiving data
        """
        try:
            ssdb = os.environ["SSDB"]
            ssdb = ssdb.strip('"')
            db_location = ssdb.split(";")
            address, port = db_location[0].split(":")

            # setup data out connection to orchestrator
            if self.cluster:
                self.db_connection= ClusterConnection()
            else:
                self.db_connection = Connection()
            self.db_connection.connect(address, port)
        except KeyError:
            raise SSConfigError("No Orchestrator found in setup!")

        if "SSKEYIN" in os.environ:
            self.get_key_prefixes = os.environ["SSKEYIN"].split(';')
        if "SSKEYOUT" in os.environ:
            self.put_key_prefix = os.environ["SSKEYOUT"]

    def protobuf_message(self, value, dtype):
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

    def _build_get_key( self, key ):
        if self.get_key_prefix:
            return "_".join((self.get_key_prefix, key))
        else:
            return key

    def _build_put_key( self, key ):
        if self.put_key_prefix:
            return "_".join((self.put_key_prefix, key))
        else:
            return key
