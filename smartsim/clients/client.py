from ..connection import Connection
from ..clusterConnection import ClusterConnection
from ..error import SmartSimError, SSConfigError, SmartSimConnectionError
from ..utils.protobuf.smartsim_protobuf_pb2 import ArrayDouble, ScalarDouble, \
    ScalarSInt64, ScalarSInt32, ArraySInt32, ArraySInt64
import pickle
import os
import numpy as np
import time

class Client:
    """The python implementation of the smartsim client used to
    communicate between the orchestrator and python programs.
    The output of any get command is a numpy object or scalar.
    """
    def __init__(self, cluster=True):
        """Initializes a SmartSim Python Client object

        :param cluster: Boolean flag indicating if a single database is
        being used (False) or a cluster of database nodes is being used
        (True).
        :type cluster: bool
        """
        self.cluster = cluster
        self.put_key_prefix = None
        self.get_key_prefix = None
        self.get_key_prefixes = []
        self.db_connection = None
        self._setup_connections()

    def get_array_nd_float64(self, key, wait=False, wait_interval=1000):
        """Get a numpy array of type float64 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArrayDouble(), True, wait, wait_interval)

    def get_array_nd_int64(self, key, wait=False, wait_interval=1000):
        """Get a numpy array of type int64 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArraySInt64(), True, wait, wait_interval)

    def get_array_nd_int32(self, key, wait=False, wait_interval=1000):
        """Get a numpy array of type int32 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArraySInt32(), True, wait, wait_interval)

    def get_scalar_float64(self, key, wait=False, wait_interval=1000):
        """Get a scalar of type float64 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: float64
        """
        return self._get_data(key, ScalarDouble(), True, wait, wait_interval)

    def get_scalar_int64(self, key, wait=False, wait_interval=1000):
        """Get a scalar of type int64 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: int64
        """
        return self._get_data(key, ScalarSInt64(), True, wait, wait_interval)

    def get_scalar_int32(self, key, wait=False, wait_interval=1000):
        """Get a scalar of type int32 from the database.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: int32
        """
        return self._get_data(key, ScalarSInt32(), True, wait, wait_interval)

    def get_exact_key_array_nd_float64(self, key, wait=False,
                                       wait_interval=1000):
        """Get a numpy array of type float64 from the database
        without key prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArrayDouble(), False, wait, wait_interval)

    def get_exact_key_array_nd_int64(self, key, wait=False,
                                     wait_interval=1000):
        """Get a numpy array of type int64 from the database
        without key prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArraySInt64(), False, wait, wait_interval)

    def get_exact_key_array_nd_int32(self, key, wait=False,
                                     wait_interval=1000):
        """Get a numpy array of type int32 from the database
        without key prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: numpy.ndarray
        """
        return self._get_data(key, ArraySInt32(), False, wait, wait_interval)

    def get_exact_key_scalar_float64(self, key, wait=False,
                                     wait_interval=1000):
        """Get a scalar of type float64 from the database
        without prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: float64
        """
        return self._get_data(key, ScalarDouble(), False, wait, wait_interval)

    def get_exact_key_scalar_int64(self, key, wait=False,
                                   wait_interval=1000):
        """Get a scalar of type int64 from the database
        without prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: int64
        """
        return self._get_data(key, ScalarSInt64(), False, wait, wait_interval)

    def get_exact_key_scalar_int32(self, key, wait=False,
                                   wait_interval=1000):
        """Get a scalar of type int32 from the database
        without key prefixing.

        :param key: key to fetch
        :type key: str
        :param wait: flag to poll database until key is present
        :type wait: bool
        :param wait_interval: time (milliseconds) between polling requests
        :type wait_interval: float
        :return: array data associated with given key
        :rtype: int32
        """
        return self._get_data(key, ScalarSInt32(), False, wait, wait_interval)

    def put_array_nd_float64(self, key, value):
        """Put a numpy array of type float64 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with float64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_float64()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('float64')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " have type float64 to use"\
                                " put_array_nd_float64().")

        self._put_data(key, value, ArrayDouble(), True)

    def put_array_nd_int64(self, key, value):
        """Put a numpy array of type int64 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with int64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int64()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int64')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " have type int64 to use put_array_nd_int64().")

        self._put_data(key, value, ArraySInt64(), True)

    def put_array_nd_int32(self, key, value):
        """Put a numpy array of type int32 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with int32
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int32()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int32')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " be of type int32 to use"\
                                " put_array_nd_int32().")

        self._put_data(key, value, ArraySInt32(), True)

    def put_scalar_float64(self, key, value):
        """Put a scalar of type float64 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: float
        """
        if not isinstance(value, float):
            raise SmartSimError("The value passed into put_scalar_float64()"\
                                " must be of type float")

        self._put_data(key, value, ScalarDouble(), True)

    def put_scalar_int64(self, key, value):
        """Put a scalar of type int64 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: int
        """
        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int64()"\
                                " must be of type int")

        self._put_data(key, value, ScalarSInt64(), True)

    def put_scalar_int32(self, key, value):
        """Put a scalar of type int32 into the database.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: int
        """
        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int32()"\
                                " must be of type int")

        self._put_data(key, value, ScalarSInt32(), True)

    def put_exact_key_array_nd_float64(self, key, value):
        """Put a numpy array of type float64 into the database
        without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with float64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_float64()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('float64')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " have type float64 to use"\
                                " put_array_nd_float64().")

        self._put_data(key, value, ArrayDouble(), False)

    def put_exact_key_array_nd_int64(self, key, value):
        """Put a numpy array of type int64 into the database
        without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with int64
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int64()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int64')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " have type int64 to use put_array_nd_int64().")

        self._put_data(key, value, ArraySInt64(), False)

    def put_exact_key_array_nd_int32(self, key, value):
        """Put a numpy array of type int32 into the database
        without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: numpy.ndarray filled with int32
        """
        if not isinstance(value, np.ndarray):
            raise SmartSimError("The value passed into put_array_nd_int32()"\
                                " must be of type numpy.ndarray.")
        if not (value.dtype == np.dtype('int32')):
            raise SmartSimError("The values inside of the numpy.ndarray must"\
                                " be of type int32 to use"\
                                " put_array_nd_int32().")

        self._put_data(key, value, ArraySInt32(), False)

    def put_exact_key_scalar_float64(self, key, value):
        """Put a scalar float64 into the database without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: float
        """
        if not isinstance(value, float):
            raise SmartSimError("The value passed into put_scalar_float64()"\
                                " must be of type float")

        self._put_data(key, value, ScalarDouble(), False)

    def put_exact_key_scalar_int64(self, key, value):
        """Put a scalar of type int64 into the database without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: int
        """
        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int64()"\
                                " must be of type int")

        self._put_data(key, value, ScalarSInt64(), False)

    def put_exact_key_scalar_int32(self, key, value):
        """Put a scalar of type int32 into the database without prefixing.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: int
        """
        if not isinstance(value, int):
            raise SmartSimError("The value passed into put_scalar_int32()"\
                                " must be of type int")

        self._put_data(key, value, ScalarSInt32(), False)

    def exact_key_exists(self, key):
        """Checks if a key is in the database"

        :param key: key to check in the database
        :type key: str
        :returns: true if the key is in the database, false otherwise
        :rtype: bool
        """
        return self.db_connection.exists(key)

    def key_exists(self, key):
        """This function checks if a key is in the database.

        This method prefixes keys to align with key prefixing
        in get and put methods.

        :param key: key to check in the database
        :type key: str
        :returns: true if the key is in the database, false otherwise
        :rtype: bool
        """
        return self.db_connection.exists(self._build_get_key(key))

    def poll_exact_key(self, key, poll_frequency=1000, num_tries=-1):
        """Poll for a key without prefixing at a specified
        frequency and specified number of times

        :param key: key to check for in the database
        :type key: str
        :param poll_frequency: the time in milliseconds beteween checks
        :type poll_frequency: float
        :param num_tires: the maximum number of tries.  If -1, unlimited
            attempts will be made.
        :type num_tries: int
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
        """Poll for a key with a specified frequency
        and specified number of times

        :param key: key to check for in the database
        :type key: str
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tires: the maximum number of tries.  If -1, unlimited
            attempts will be made.
        :type num_tries: int
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

    def poll_key_and_check_scalar_float64(self, key, value, poll_frequency=1000,
                                          num_tries=-1):
        """Poll database for existence of a key and if it is exists check gainst value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: float
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found
            after specified number of tries, otherwise true.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                               self.get_scalar_float64,
                                               self.key_exists,
                                               poll_frequency, num_tries)

    def poll_key_and_check_scalar_int64(self, key, value, poll_frequency=1000,
                                        num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: int
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found
            after specified number of tries, otherwise true.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                               self.get_scalar_int64,
                                               self.key_exists,
                                               poll_frequency, num_tries)

    def poll_key_and_check_scalar_int32(self, key, value, poll_frequency=1000,
                                        num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: float
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: int
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found after
            specified number of tires.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                               self.get_scalar_int32,
                                               self.key_exists,
                                               poll_frequency, num_tries)

    def poll_exact_key_and_check_scalar_float64(self, key, value,
                                           poll_frequency=1000, num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: float
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found
            after specified number of tries, otherwise true.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                            self.get_exact_key_scalar_float64,
                                            self.exact_key_exists,
                                            poll_frequency, num_tries)

    def poll_exact_key_and_check_scalar_int64(self, key, value,
                                        poll_frequency=1000, num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: int
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found
            after specified number of tries, otherwise true.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                            self.get_exact_key_scalar_int64,
                                            self.exact_key_exists,
                                            poll_frequency, num_tries)

    def poll_exact_key_and_check_scalar_int32(self, key, value,
                                        poll_frequency=1000, num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: float
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: int
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found
            after specified number of tries, otherwise true.
        :rtype: bool
        """
        return self._poll_key_and_check_scalar(key, value,
                                            self.get_exact_key_scalar_int32,
                                            self.exact_key_exists,
                                            poll_frequency, num_tries)

    def _poll_key_and_check_scalar(self, key, value,
                                   get_function, exists_function,
                                   poll_frequency=1000, num_tries=-1):
        """Poll database for existence of a key and if it is exists check
        against value

        :param key: key to check for in the database
        :type key:
        :param value: value to compare to
        :type value: float
        :get_function: pointer to Client function for retrieving scalar
        :type get_function: function pointer
        :exists_function: pointer to Client function for checking key
        :type exists_function: function pointer
        :param poll_frequency: the time in milliseconds beteween tries
        :type poll_frequency: float
        :param num_tries: the maximum number of tries.  If <0, unlimited
            attempts will be made.
        :type num_tries: int
        :returns: False if the key with matching value could not be found after
            specified number of tires.
        :rtype: bool
        """
        matched_value = False
        current_value = None

        while not num_tries == 0:
            if exists_function(key):
                current_value = get_function(key)
                if current_value == value:
                    num_tries = 0
                    matched_value = True
            if not matched_value:
                time.sleep(poll_frequency/1000.0)
            if num_tries>0:
                num_tries -= 1

        return matched_value

    def set_data_source( self, prefix):
        """Set the prefix used for key queries from the database. The prefix is
        checked to ensure that it is a valid data source

        :param prefix: prefix used for key queries
        :type prefix: str
        """
        if prefix in self.get_key_prefixes:
            self.get_key_prefix = prefix
        else:
            raise SmartSimError(f"{prefix} has not been registered")

    def _get_data(self, key, pb_message, add_prefix, wait=False,
                  wait_interval=1000):
        """Get the value associated with a key.  Wait implies that
        the request should be made until a key by that name is stored
        in the connection.

        :param key: key of the value being retrieved
        :type key: str
        :param dtype: the numpy data type of the serialized data (e.g. float64)
        :type dtype: str
        :param add_prefix: if true, add a prefix to the key
        :type add_prefix: bool
        :param wait: flag for polling connection until value appears
        :type wait: bool
        :param wait_interval: milliseconds to wait between polling requests
        :type wait_interval: float
        :returns: data associated with the key
        :rtype: float, int, or np.ndarray
        """
        if add_prefix:
            query_key = self._build_get_key(key)
        else:
            query_key = key

        if wait:
            self.poll_exact_key(query_key, poll_frequency=wait_interval,
                                num_tries=-1)


        data_bytes = self.db_connection.get(query_key, wait=wait,
                                            wait_interval=wait_interval)
        data = self._deserialize(pb_message, data_bytes)

        return data

    def _put_data(self, key, value, pb_message, add_prefix):
        """Put the value with the given key.

        :param key: key of the value being put
        :type key: str
        :param value: value being put
        :type value: int, float, or numpy.ndarray
        :param add_prefix: if true, add a prefix to the key
        :type add_prefix: bool
        :param pb_message: protobuf message object
        :type pb_message: google.protobuf.pb_message
        :param add_prefix: if true, potentially add a prefix to the key
        :type add_prefix: bool
        """
        if type(key) != str:
            raise SmartSimError("Key must be of string type")
        if not self.db_connection:
            raise SmartSimError("Connection to database was not setup"\
                                " at runtime")

        value_bytes = self._serialize(pb_message, value)
        send_key = key
        if self.put_key_prefix and add_prefix:
            send_key = "_".join((self.put_key_prefix, key))
        self.db_connection.send(send_key, value_bytes)

    def _serialize(self, pb_message, value):
        """Serializes value using protocol buffer

        :param pb_message: protobuf message object
        :type pb_message: google.protobuf.message
        :param value: data to serialize
        :type value: numpy.ndarray or scalar
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

    def _deserialize(self, pb_message, data_bytes):
        """Unpacks the serialized protocol buffer into a numpy.ndarray

        :param pb_message: protobuf message object to decode bytes
        :type pb_message: google.protobuf.message
        :param data_byes: the serialized data
        :type value: string
        :returns: protobuf message data
        :rtype: numpy.ndarray, float, or int
        """
        pb_message.ParseFromString(data_bytes)
        if hasattr(pb_message, 'dimension'):
            dimensions = pb_message.dimension
            data = np.reshape(pb_message.data, dimensions)
        else:
            data = pb_message.data
        return data

    def _setup_connections(self):
        """Retrieve the environment variables to set up connections
        for this client.
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

    def _build_get_key( self, key ):
        """Prefix the key with the appropriate value
        :param key: the key without prefix
        :type key: str
        :return: a prefixed key
        :rtype: str
        """
        if self.get_key_prefix:
            return "_".join((self.get_key_prefix, key))
        else:
            return key

    def _build_put_key( self, key ):
        """Prefix the key with the appropriate value
        :param key: the key without prefix
        :type key: str
        :return: a prefixed key
        :rtype: str
        """
        if self.put_key_prefix:
            return "_".join((self.put_key_prefix, key))
        else:
            return key
