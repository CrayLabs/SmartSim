from smartsim import Client
from sys import float_info

def put_get_exact_key_scalar(key, putFunction, getFunction, value, source):
    """This function puts a scalar into the database,
    retrieves the scalar from the database, and then
    compares the sent and received scalars for equality.
    Keys are not prefixed when sent to the database.

    :param key: key to use
    :type key: str
    :param putFunction: client object put function
    :type putFunction: function pointer
    :param getFunction: client object get function
    :type getFunction: function pointer
    :param value: The value to put into the database
    :type value: float or int
    :param source: name of the data source to set in client
    :type source: str
    """
    client = Client(cluster=True)

    if(source):
        client.set_data_source(source)

    print(f"Starting put with key {key}", flush=True)
    putFunction(client, key, value)
    print(f"Finished put with key {key}", flush=True)

    print(f"Starting get with key {key}", flush=True)
    result = getFunction(client, key)
    print(f"Finished get with key {key}", flush=True)

    assert(value==result)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source", type=str, default="")
    args = argparser.parse_args()

    # Float64
    put_function = Client.put_exact_key_scalar_float64
    get_function = Client.get_exact_key_scalar_float64
    # Float64 (max value)
    key = "put_get_exact_key_dbl_max"
    value = float_info.max
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)
    # Float64 (min value)
    key = "put_get_exact_key_dbl_min"
    value = float_info.min
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)

    #Int64
    put_function = Client.put_exact_key_scalar_int64
    get_function = Client.get_exact_key_scalar_int64
    # Int64 (max value)
    key = "put_get_exact_key_i64_max"
    value = 9223372036854775807
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)
    # Int64 (min value)
    key = "put_get_exact_key_i64_min"
    value = -9223372036854775808
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)

    #Int32
    put_function = Client.put_exact_key_scalar_int32
    get_function = Client.get_exact_key_scalar_int32
    # Int32 (max value)
    key = "put_get_exact_key_i32_max"
    value = 2147483647
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)
    # Int32 (min value)
    key = "put_get_exact_key_i32_min"
    value = -2147483648
    put_get_exact_key_scalar(key, put_function, get_function, value,
                            args.source)