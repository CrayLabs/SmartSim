
import time
import pickle
from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client(cluster=True)

    def train_loop(self):
        i = 0
        avg_one_way_time = 0
        while True:
            data = self.client.get_data(str(i), wait=True)
            receive_time = time.time()
            data = pickle.loads(data)
            send_time = data[1]
            avg_one_way_time += abs(receive_time - send_time)
            print("Receiving data for key", str(i))
            i+=1
            if i == 20:
                break
        print("Average One Way Latency:", str(avg_one_way_time/i), flush=True)

if __name__ == "__main__":
    tn = Node()
    tn.client.setup_connections()
    tn.train_loop()