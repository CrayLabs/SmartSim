from smartsim import Client
import sys
import time

if __name__ == "__main__":

    client = Client(cluster=True)

    client.put_scalar_float64("STATUS",0.0)
    time.sleep(2)
    client.put_scalar_float64("STATUS",5.5)
