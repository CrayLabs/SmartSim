
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

def run_inference_loop(data_size, num_packets, client):

    i = 0
    prep_times = 0
    total_inf_loop_ave = 0
    while i < int(num_packets):

        # get data to send to node
        data = create_data(i, data_size)

        # send to node
        start_time = time.time()
        obj = pickle.dumps((data, start_time))
        print("sending data for key", str(i), flush=True)
        client.send_big_data(str(i), obj)
        send_time = time.time()
        print("Data sent at: ", str(send_time), flush=True)

        # calculate time to pack bytes
        prep_time = (send_time - start_time)
        prep_times += prep_time

        # get the data back from the training node
        # dont include pickling time in calculations
        processed_obj = client.get_data(str(i), wait=True)
        stop_time = time.time()
        total_inference_loop_time = (stop_time - start_time)
        total_inf_loop_ave += total_inference_loop_time

        # ensure data made it back the same
        assert(processed_obj == obj)

        i+=1
    print("Avg Prep Time: ", str(prep_times/num_packets))
    print("Ave Inference Loop Time: ", str(total_inf_loop_ave/num_packets))


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=str, default="(20,)")
    argparser.add_argument("--num_packets", type=int, default=20)
    args = argparser.parse_args()

    # setup client and begin sending data
    client = Client()
    client.setup_connections()
    run_inference_loop(args.size, args.num_packets, client)
