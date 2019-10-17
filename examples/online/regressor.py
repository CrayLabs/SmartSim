
import time
import pickle
from sklearn.neural_network import MLPRegressor

from smartsim import Client # The smartsim user client library

class TrainingNode():

    def __init__(self):
        self.client = Client()

    def train_loop(self):
        i = 0
        model = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter=10000,
                            activation='logistic', verbose='True', learning_rate='adaptive')
        while True:
            while self.client.get_data(str(i)) == None:
                    time.sleep(2) # be kind to system
            X, y = pickle.loads(self.client.get_data(str(i)))
            
            print("Training model on data", str(i), flush=True)
            model.fit(X, y)

            # increment the time-step
            i+=1
            if i == 20:
                break # end the training session
                


if __name__ == "__main__":

    tn = TrainingNode()
    tn.client.setup_connections()
    tn.train_loop()
