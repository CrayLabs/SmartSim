
import time
import numpy as np
import zmq
import pickle

from smartsim.connection import Connection


def simulate(seed):
    
    # mimic computationally expensive kernel
    time.sleep(1)
    
    # conduct simulation of a simple 2D function
    # f(x) = x**2
    np.random.seed(seed)
    n = 20
    x = np.random.uniform(-15, 15, size = n)
    y = x**2 + 2*np.random.randn(n, )
    X = np.reshape(x ,[n, 1]) 
    y = np.reshape(y ,[n ,])
        
    return(X, y)

def run_simulations(steps, client):

    i = 0
    while i < int(steps):
        data = simulate(i)
        obj = pickle.dumps(data)
        print("sending data for key", str(i), flush=True)
        client.send(str(i), obj)
        i+=1
        


if __name__ == "__main__":

    # server connection provided as an argument
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--port", type=str, default="6379")
    argparser.add_argument("--addr", type=str, default="nid00110")
    argparser.add_argument("--db", type=int, default=0)
    argparser.add_argument("--steps", type=int, default=20)
    args = argparser.parse_args()
    
    client = Connection()
    client.connect(args.addr, args.port, args.db)
    run_simulations(args.steps, client)
    

