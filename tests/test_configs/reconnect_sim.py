import numpy as np

from smartsim import Client


def create_data(seed, size):

    np.random.seed(seed)
    x = np.random.uniform(-15.0, 15.0, size=size)
    return x

if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cluster", default=False, action='store_true')
    args = argparser.parse_args()

    client = None
    if args.cluster:
        client = Client(cluster=True)
    else:
        client = Client(cluster=False)

    for i in range(0,5):
        data = create_data(i, 100)
        client.put_array_nd_float64(str(i), data)
        for d in data:
            print(d)
