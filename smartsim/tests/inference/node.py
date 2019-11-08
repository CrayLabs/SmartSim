
import time
import pickle
from smartsim import Client

class TrainingNode():

    def __init__(self):
        self.client = Client()

    def train_loop(self):
        i = 0
        avg_train_path_time = 0
        while True:
            # recieve data from the simulation
            obj = self.client.get_data(str(i), wait=True)

            # get the time and compare to time sent
            recieve_time = time.time()
            data = pickle.loads(obj)
            start_time = data[1]
            avg_train_path_time += abs(recieve_time - start_time)
            print("Recieving data for key", str(i))
            print("Data recieved at: " , str(recieve_time), flush=True)

            # send back immediately to test inference loop overhead
            self.client.send_big_data(str(i), obj)

            i+=1
            if i == 20:
                break
        print("Average Train Path latency:", str(avg_train_path_time/i), flush=True)


if __name__ == "__main__":

    tn = TrainingNode()
    tn.client.setup_connections()
    tn.train_loop()