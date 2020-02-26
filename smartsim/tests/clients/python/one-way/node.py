
from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client(cluster=True)

    def train_loop(self):
        i = 0
        while i <= 19:
            data = self.client.get_data(str(i), "float64", wait=True)
            print("Receiving data for key", str(i), flush=True)
            i+=1

if __name__ == "__main__":
    tn = Node()
    tn.client.setup_connections()
    tn.train_loop()