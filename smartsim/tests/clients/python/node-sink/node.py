
import time
import pickle
from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client(cluster=True)

    def train_loop(self):
        i = 0
        while i <= 4:
            data = self.client.get_array_nd_float64(str(i), wait=True)
            sim_1_data = data["sim_1"]
            sim_2_data = data["sim_2"]
            receive_time = time.time()
            assert(len(data.keys()) == 2)
            print("Receiving data for key", str(i))
            i+=1

if __name__ == "__main__":
    tn = Node()
    tn.client.setup_connections()
    tn.train_loop()
