from smartsim import Client

if __name__ == "__main__":
    client = Client(cluster=True)
    client.setup_connections()
    assert(client.poll_key_and_check_scalar_float64("STATUS", 5.5, poll_frequency=500))
