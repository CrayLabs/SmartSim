from smartsim import Client
import sys
import time

if __name__ == "__main__":

    client = Client(cluster=True)
    client.setup_connections()

    client.put_scalar_int64("STATUS",0)
    time.sleep(2)
    client.put_scalar_int64("STATUS",5)
