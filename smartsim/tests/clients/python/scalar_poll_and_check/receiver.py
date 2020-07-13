from smartsim import Client
from sys import float_info

def poll_and_check_scalar(key_suffix, pollFunction, value, source):
    """This polls for a key in the database and checks
    that its value matches the expected value.

    :param key_suffix: suffix added to the key
    :type key_suffix: str
    :param pollFunction: client object poll and check function
    :type pollFunction: function pointer
    :param value: The value to put into the database
    :type value: float or int
    :param source: name of the data source to set in client
    :type source: str
    """
    client = Client(cluster=True)

    if(source):
        client.set_data_source(source)

    key = f"poll_check_scalar{key_suffix}"
    print(f"Starting poll and check with key {key}", flush=True)
    match_found = pollFunction(client, key, value, poll_frequency=100,
                               num_tries=30)
    print(f"Finished poll and check with key {key}", flush=True)
    assert(match_found)

    print(f"Starting poll with key {key}", flush=True)
    match_found = client.poll_key(key, poll_frequency=100,
                                  num_tries=30)
    print(f"Finished poll with key {key}", flush=True)
    assert(match_found)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source", type=str, default="")
    args = argparser.parse_args()

    source = args.source

    # Float64
    poll_check_function = Client.poll_key_and_check_scalar_float64
    # Float64 (max value)
    suffix = "_dbl_max"
    value = float_info.max
    poll_and_check_scalar(suffix, poll_check_function, value, source)
    # Float64 (min value)
    suffix = "_dbl_min"
    value = float_info.min
    poll_and_check_scalar(suffix, poll_check_function, value, source)

    #Int64
    poll_check_function = Client.poll_key_and_check_scalar_int64
    # Int64 (max value)
    suffix = "_i64_max"
    value = 9223372036854775807
    poll_and_check_scalar(suffix, poll_check_function, value, source)
    # Int64 (min value)
    suffix = "_i64_min"
    value = -9223372036854775808
    poll_and_check_scalar(suffix, poll_check_function, value, source)

    #Int32
    poll_check_function = Client.poll_key_and_check_scalar_int32
    # Int32 (max value)
    suffix = "_i32_max"
    value = 2147483647
    poll_and_check_scalar(suffix, poll_check_function, value, source)
    # Int32 (min value)
    suffix = "_i32_min"
    value = -2147483648
    poll_and_check_scalar(suffix, poll_check_function, value, source)