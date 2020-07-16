from smartsim import Client
from sys import float_info
import random

def put_get_scalar(key_suffix, putFunction, getFunction, value, source):
    """This function puts a scalar into the database,
    retrieves the scalar from the database, and then
    compares the sent and received scalars for equality.

    :param key_suffix: suffix added to the key
    :type key_suffix: str
    :param putFunction: client object put function
    :type putFunction: function pointer
    :param getFunction: client object get function
    :type getFunction: function pointer
    :param value: The value to put into the database
    :type value: float or int
    :param source: name of the data source to set in client
    :type source: str
    """
    client = Client(cluster=False)

    if(source):
        client.set_data_source(source)

    key = f"put_get_scalar{key_suffix}"
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
    put_function = Client.put_scalar_float64
    get_function = Client.get_scalar_float64
    # Float64 (max value)
    suffix = "_dbl_max"
    value = float_info.max
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Float64 (min value)
    suffix = "_dbl_min"
    value = float_info.min
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Float64 (random value)
    suffix = "_dbl_rand"
    value = random.random()
    put_get_scalar(suffix, put_function, get_function, value, args.source)

    #Int64
    put_function = Client.put_scalar_int64
    get_function = Client.get_scalar_int64
    # Int64 (max value)
    suffix = "_i64_max"
    value = 9223372036854775807
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Int64 (min value)
    suffix = "_i64_min"
    value = -9223372036854775808
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Int64 (random value)
    suffix = "_i64_rand"
    value = random.randint(-9223372036854775808, 9223372036854775807)
    put_get_scalar(suffix, put_function, get_function, value, args.source)

    #Int32
    put_function = Client.put_scalar_int32
    get_function = Client.get_scalar_int32
    # Int32 (max value)
    suffix = "_i32_max"
    value = 2147483647
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Int32 (min value)
    suffix = "_i32_min"
    value = -2147483648
    put_get_scalar(suffix, put_function, get_function, value, args.source)
    # Int32 (rand value)
    suffix = "_i32_rand"
    value = random.randint(-2147483648, 2147483647)
    put_get_scalar(suffix, put_function, get_function, value, args.source)
