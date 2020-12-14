import argparse
import pickle

import numpy as np
import redis


def send_data(key):
    client = redis.Redis(host="localhost", port=6780)
    data = np.random.uniform(0.0, 10.0, size=(5000))
    serialized_data = pickle.dumps(data)
    client.set(key, serialized_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--name", type=str, default="model")
    args = parser.parse_args()

    # send data in iterations
    for i in range(args.iters):
        key = "key_" + args.name + "_" + str(i)
        send_data(key)