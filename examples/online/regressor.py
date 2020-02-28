
import time
import pickle
from sklearn.neural_network import MLPRegressor

from smartsim import Client # The smartsim user client library

class TrainingNode():

    def __init__(self):
        self.client = Client()

    def train_loop(self):
        i = 0
        # warm start to use previous weights as initialization
        model = MLPRegressor(alpha=0.001, hidden_layer_sizes=(10,),
                             max_iter=5000, activation='logistic',
                             verbose='True', learning_rate='adaptive',
                             warm_start=True, tol=-4000)
        while i <= 19:
            X = self.client.get_data(str(i) + "_X", "float64", wait=True)
            y = self.client.get_data(str(i) + "_y", "float64", wait=True)
            print("-------------------------------")
            print("Training on time step ", str(i))
            print("-------------------------------")
            model.fit(X, y)
            i += 1

if __name__ == "__main__":

    tn = TrainingNode()
    tn.client.setup_connections()
    tn.train_loop()
