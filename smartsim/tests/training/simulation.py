
import time
import numpy as np
import pickle

from smartsim import Client
from ast import literal_eval
import sys

def create_data(seed, size):

    np.random.seed(seed)
    x = np.random.uniform(-15, 15, size=literal_eval(size))
    return x

def run_simulations(data_size, num_packets, client):

    i = 0
    prep_times = 0
    while i < int(num_packets):
        data = create_data(i, data_size)
        start_time = time.time()
        obj = pickle.dumps((data, start_time))
        print("sending data for key", str(i), flush=True)
        client.send_data(str(i), obj)
        send_time = time.time()
        print("Data sent at: ", str(send_time), flush=True)
        prep_time = (send_time - start_time)
        prep_times += prep_time
        i+=1
    print("Avg Prep Time: ", str(prep_times/num_packets))


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=str, default="(20,)")
    argparser.add_argument("--num_packets", type=int, default=20)
    args = argparser.parse_args()

    # setup client and begin sending data
    client = Client()
    client.setup_connections()
    run_simulations(args.size, args.num_packets, client)
