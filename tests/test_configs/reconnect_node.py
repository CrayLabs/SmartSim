import numpy as np

from smartsim import Client

if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cluster", default=False, action="store_true")
    args = argparser.parse_args()

    client = None
    if args.cluster:
        client = Client(cluster=True)
    else:
        client = Client(cluster=False)

    for i in range(0, 5):
        data = client.get_array_nd_float64(str(i))
        for d in data:
            print(d)
