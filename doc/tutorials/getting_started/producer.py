import numpy as np
import argparse
import time
import os

from smartredis import Client, ConfigOptions

parser = argparse.ArgumentParser(description="SmartRedis ensemble producer process.")
parser.add_argument("--redis-port")
args = parser.parse_args()

time.sleep(2)
address = "127.0.0.1:" + str(args.redis_port)
os.environ["SSDB"] = address
c = Client(None, logger_name="SmartSim")

data = np.random.rand(1, 1, 3, 3)
c.put_tensor("product", data)