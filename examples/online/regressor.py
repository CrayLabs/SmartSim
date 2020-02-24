
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
        while True:
            data = self.client.get_data(str(i), wait=True)
            X, y = pickle.loads(data)
            print("-------------------------------")
            print("Training on time step ", str(i))
            print("-------------------------------")
            time.sleep(1)
            model.fit(X, y)

            i+=1
            if i == 20:
                break # end the training session

if __name__ == "__main__":

    tn = TrainingNode()
    tn.client.setup_connections()
    tn.train_loop()
