import numpy as np

from smartsim import Client
from ast import literal_eval
import sys

def create_data(seed, size):

    np.random.seed(seed)
    x = np.random.uniform(-15.0, 15.0, size=literal_eval(size))
    return x

def run_full_loop(data_size, num_packets, client):

    i = 0
    while i < int(num_packets):
        # get data to send to node
        data = create_data(i, data_size)
        print("Sending data for key", str(i), flush=True)
        client.send_data(str(i), data)

        # get the data back from the node
        processed_data = client.get_data(str(i), "float64", wait=True)

        # ensure data made it back the same
        assert(processed_data.all() == data.all())
        i+=1


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=str, default="(20,)")
    argparser.add_argument("--num_packets", type=int, default=20)
    args = argparser.parse_args()

    # setup client and begin sending data
    client = Client()
    client.setup_connections()
    run_full_loop(args.size, args.num_packets, client)
