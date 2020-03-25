import numpy as np

from smartsim import Client
from ast import literal_eval
import sys

def create_data(seed, size):

    np.random.seed(seed)
    x = np.random.uniform(-15.0, 15.0, size=literal_eval(size))
    return x

def run_simulations(data_size, num_packets, client):

    i = 0
    prep_times = 0
    while i < int(num_packets):
        data = create_data(i, data_size)
        print("Sending data for key", str(i), flush=True)
        client.put_array_nd_float64(str(i), data)
        r_data = client.get_array_nd_float64(str(i))
        assert(np.array_equal(data, r_data))
        i+=1

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=str, default="(20,)")
    argparser.add_argument("--num_packets", type=int, default=20)
    args = argparser.parse_args()

    # setup client and begin sending data
    client = Client(cluster=True)
    client.setup_connections()
    run_simulations(args.size, args.num_packets, client)
