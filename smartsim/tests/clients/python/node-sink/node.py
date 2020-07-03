
import time
import pickle
from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client(cluster=True)

    def train_loop(self):
        i = 0
        while i <= 4:
            print("Receiving data for key", str(i))
            self.client.set_data_source("sim_1")
            sim_1_data = self.client.get_array_nd_float64(str(i), wait=True)
            self.client.set_data_source("sim_2")
            sim_2_data = self.client.get_array_nd_float64(str(i), wait=True)
            receive_time = time.time()
            i+=1

if __name__ == "__main__":
    tn = Node()
    tn.train_loop()
