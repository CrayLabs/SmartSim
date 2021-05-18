import ray
from ray import tune
import ray.util
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="PPO Tune Example")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster.")
args = parser.parse_args()

ray.util.connect(args.ray_address.split(':')[0]+":10001")
# print("connected")
#ray.init(address=args.ray_address, _redis_password=args.redis_password)
print("initialized")
tune.run(
    "PPO",
    stop={"episode_reward_max": 200},
    config={
        "framework": "torch",
        "env": "CartPole-v0",
        "num_gpus": 0,
        "lr": tune.grid_search(np.linspace (0.001, 0.01, 100).tolist()),
        "log_level": "ERROR",
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
    },
    local_dir="/lus/scratch/arigazzi/ray_local/",
    verbose=3,
    fail_fast=True,
)