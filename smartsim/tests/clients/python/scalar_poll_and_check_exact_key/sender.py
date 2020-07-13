from smartsim import Client
from sys import float_info

def put_scalar(key_suffix, putFunction, value):
    """This function puts a scalar into the database.

    :param key_suffix: suffix added to the key
    :type key_suffix: str
    :param putFunction: client object put function
    :type putFunction: function pointer
    :param value: The value to put into the database
    :type value: float or int
    """
    client = Client(cluster=True)

    key = f"poll_check_exact_key_scalar{key_suffix}"
    print(f"Starting put with key {key}", flush=True)
    putFunction(client, key, value)
    print(f"Finished put with key {key}", flush=True)

if __name__ == "__main__":
    # Float64
    put_function = Client.put_exact_key_scalar_float64
    # Float64 (max value)
    suffix = "_dbl_max"
    value = float_info.max
    put_scalar(suffix, put_function, value)
    # Float64 (min value)
    suffix = "_dbl_min"
    value = float_info.min
    put_scalar(suffix, put_function, value)

    #Int64
    put_function = Client.put_exact_key_scalar_int64
    # Int64 (max value)
    suffix = "_i64_max"
    value = 9223372036854775807
    put_scalar(suffix, put_function, value)
    # Int64 (min value)
    suffix = "_i64_min"
    value = -9223372036854775808
    put_scalar(suffix, put_function, value)

    #Int32
    put_function = Client.put_exact_key_scalar_int32
    # Int32 (max value)
    suffix = "_i32_max"
    value = 2147483647
    put_scalar(suffix, put_function, value)
    # Int32 (min value)
    suffix = "_i32_min"
    value = -2147483648
    put_scalar(suffix, put_function, value)