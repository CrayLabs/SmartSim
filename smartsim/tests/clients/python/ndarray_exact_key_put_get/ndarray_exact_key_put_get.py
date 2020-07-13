from smartsim import Client
import numpy as np

def fill_nd_array_w_floating_point(array):
    """This function fills an ndarray with random
    floating point values.

    :param array: the array to be filled
    :type array: numpy.ndarray
    """
    with np.nditer(array, order='C', op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.random.uniform(-100.0, 100.0)

def fill_nd_array_w_integer(array):
    """This function fills an ndarray with random
    integer values.

    :param array: the array to be filled
    :type array: numpy.ndarray
    """
    with np.nditer(array, order='C', op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.random.randint(-100, 100)

def put_get_array_exact_key(dims, dtype, key, putFunction,
                            getFunction, fillFunction, source):
    """This function puts an ndarray into the database,
    retrieves the ndarray from the database, and then
    compares the sent and received arrays for equality.
    Keys are not prefixed when sent to the database.

    :param dims: the dimensions of the array to send
    :type dims: tuple of integers
    :param dtype: the numpy type used by the ndarray
    :type dtype: numpy.dtype
    :param key: key to use
    :type key: str
    :param putFunction: client object put function
    :type putFunction: function pointer
    :param getFunction: client object get function
    :type getFunction: function pointer
    :param fillFunction: function used to fill ndarray with values
    :type fillFunction: function pointer
    :param source: name of the data source to set in client
    :type source: str
    """
    client = Client(cluster=True)

    if(source):
        client.set_data_source(source)

    array = np.ndarray(dims,dtype=dtype)
    fillFunction(array)

    print(f"Starting put with key {key}", flush=True)
    putFunction(client, key, array)
    print(f"Finished put with key {key}", flush=True)

    print(f"Starting get with key {key}", flush=True)
    result = getFunction(client, key)
    print(f"Finished get with key {key}", flush=True)

    assert(np.array_equal(array,result))

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=20)
    argparser.add_argument("--source", type=str, default="")

    args = argparser.parse_args()

    put_function = Client.put_exact_key_array_nd_float64
    get_function = Client.get_exact_key_array_nd_float64
    fill_function = fill_nd_array_w_floating_point
    dtype = np.dtype(np.float64)
    # 1D Float64
    key = "put_get_exact_key_dbl_1D"
    put_get_array_exact_key((args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 2D Float64
    key = "put_get_exact_key_dbl_2D"
    put_get_array_exact_key((args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 3D Float64
    key = "put_get_exact_key_dbl_3D"
    put_get_array_exact_key((args.size, args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)

    put_function = Client.put_exact_key_array_nd_int64
    get_function = Client.get_exact_key_array_nd_int64
    fill_function = fill_nd_array_w_integer
    dtype = np.dtype(np.int64)
    # 1D Int64
    key = "put_get_exact_key_i64_1D"
    put_get_array_exact_key((args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 2D Int64
    key = "put_get_exact_key_i64_2D"
    put_get_array_exact_key((args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 3D Int64
    key = "put_get_exact_key_i64_3D"
    put_get_array_exact_key((args.size, args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)

    put_function = Client.put_exact_key_array_nd_int32
    get_function = Client.get_exact_key_array_nd_int32
    fill_function = fill_nd_array_w_integer
    dtype = np.dtype(np.int32)
    # 1D Int32
    key = "put_get_exact_key_i32_1D"
    put_get_array_exact_key((args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 2D Int32
    key = "put_get_exact_key_i32_2D"
    put_get_array_exact_key((args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)
    # 3D Int32
    key = "put_get_exact_key_i32_3D"
    put_get_array_exact_key((args.size, args.size, args.size),
                            dtype, key, put_function,
                            get_function, fill_function, args.source)


