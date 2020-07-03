from smartsim import Client

if __name__ == "__main__":
    client = Client(cluster=True)
    assert(client.poll_key_and_check_scalar_int64("STATUS", 5, poll_frequency=500))
