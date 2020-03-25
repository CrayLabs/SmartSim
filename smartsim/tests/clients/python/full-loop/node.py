from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client()

    def train_loop(self):
        i = 0
        while i <= 19:
            # recieve data from the simulation
            data = self.client.get_array_nd_float64(str(i), wait=True)
            print("Recieving data for key", str(i), flush=True)

            self.client.put_array_nd_float64(str(i), data)
            i+=1


if __name__ == "__main__":

    tn = Node()
    tn.client.setup_connections()
    tn.train_loop()
