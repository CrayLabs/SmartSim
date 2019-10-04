

from smartsim.connection import Connection
from sklearn.neural_network import MLPRegressor
import time
import pickle


class TrainingNode():
    
    def __init__(self):
        self.conn = Connection()

    def train_loop(self):
        i = 0
        model = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter=50000, 
                            activation='logistic', verbose='True', learning_rate='adaptive')
        while True:
            while self.conn.get(str(i)) == None:
                    time.sleep(2) # be kind to system    
            X, y = pickle.loads(self.conn.get(str(i)))
            model.partial_fit(X, y)
            print("Training model on data", str(i), flush=True)
            i+=1
        
        
    
    
if __name__ == "__main__":
    
    # server connection provided as an argument
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--port", type=str, default="6379")
    argparser.add_argument("--addr", type=str, default="nid00110")
    argparser.add_argument("--db", type=int, default=0)
    args = argparser.parse_args()
    
    tn = TrainingNode()
    tn.conn.connect(args.addr, args.port, args.db)
    tn.train_loop()