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

def put_get_array(dims, dtype, key_suffix, putFunction,
                  getFunction, fillFunction, source):
    """This function puts an ndarray into the database,
    retrieves the ndarray from the database, and then
    compares the sent and received arrays for equality.

    :param dims: the dimensions of the array to send
    :type dims: tuple of integers
    :param dtype: the numpy type used by the ndarray
    :type dtype: numpy.dtype
    :param key_suffix: suffix added to the key
    :type key_suffix: str
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

    key = f"put_get_nd_array{key_suffix}"
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

    put_function = Client.put_array_nd_float64
    get_function = Client.get_array_nd_float64
    fill_function = fill_nd_array_w_floating_point
    dtype = np.dtype(np.float64)
    suffix = "_dbl"
    # 1D Float64
    put_get_array((args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 2D Float64
    put_get_array((args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 3D Float64
    put_get_array((args.size, args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)

    put_function = Client.put_array_nd_int64
    get_function = Client.get_array_nd_int64
    fill_function = fill_nd_array_w_integer
    dtype = np.dtype(np.int64)
    suffix = "_i64"
    # 1D Int64
    put_get_array((args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 2D Int64
    put_get_array((args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 3D Int64
    put_get_array((args.size, args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)

    put_function = Client.put_array_nd_int32
    get_function = Client.get_array_nd_int32
    fill_function = fill_nd_array_w_integer
    dtype = np.dtype(np.int32)
    suffix = "_i32"
    # 1D Int32
    put_get_array((args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 2D Int32
    put_get_array((args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)
    # 3D Int32
    put_get_array((args.size, args.size, args.size),
                  dtype, suffix, put_function,
                  get_function, fill_function,
                  args.source)


