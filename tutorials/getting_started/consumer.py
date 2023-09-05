import argparse
import os
from smartredis import Client, ConfigOptions

parser = argparse.ArgumentParser(description="SmartRedis ensemble consumer process.")
parser.add_argument("--redis-port")
args = parser.parse_args()

# get model and set into database
address = "127.0.0.1:" + str(args.redis_port)
os.environ["SSDB"] = address
c = Client(None, logger_name="SmartSim")


# Incoming entity prefixes are stored as a comma-separated list
# in the env variable SSKEYIN
keyin = os.getenv("SSKEYIN")
data_sources = keyin.split(",")
data_sources.sort()

for key in data_sources:
    c.set_data_source(key)
    input_exists = c.poll_tensor("product", 100, 100)
    db_tensor = c.get_tensor("product")
    print(f"Tensor for {key} is:", db_tensor)