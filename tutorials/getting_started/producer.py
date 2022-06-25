import numpy as np
import argparse
import time

from smartredis import Client

parser = argparse.ArgumentParser(description="SmartRedis ensemble producer process.")
parser.add_argument("--redis-port")
args = parser.parse_args()

time.sleep(2)
c = Client(address="127.0.0.1:"+str(args.redis_port), cluster=False)
data = np.random.rand(1, 1, 3, 3)
c.put_tensor("product", data)