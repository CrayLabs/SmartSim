
import time
import pickle
from smartsim import Client

class Node():

    def __init__(self):
        self.client = Client()

    def train_loop(self):
        i = 0
        avg_one_way_time = 0
        while True:
            # recieve data from the simulation
            obj = self.client.get_data(str(i), wait=True)

            # get the time and compare to time sent
            recieve_time = time.time()
            data = pickle.loads(obj)
            start_time = data[1]
            avg_one_way_time += abs(recieve_time - start_time)
            print("Recieving data for key", str(i))

            # send back immediately to test full loop overhead
            self.client.send_data(str(i), obj)

            i+=1
            if i == 20:
                break
        print("Average One Way latency:", str(avg_one_way_time/i), flush=True)


if __name__ == "__main__":

    tn = Node()
    tn.client.setup_connections()
    tn.train_loop()