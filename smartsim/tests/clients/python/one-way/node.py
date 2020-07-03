from smartsim import Client
import time

class Node():

    def __init__(self):
        self.client = Client(cluster=True)

    def train_loop(self):
        i = 0
        while i <= 4:
            self.client.poll_key(str(i))
            print("Found key " + str(i))
            data = self.client.get_array_nd_float64(str(i))
            print("Receiving data for key", str(i), flush=True)
            i+=1

if __name__ == "__main__":
    tn = Node()
    tn.train_loop()
