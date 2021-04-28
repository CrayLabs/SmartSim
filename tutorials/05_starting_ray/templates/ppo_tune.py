print("Ready to import ray")
import ray
from ray import tune
import time
import numpy as np
from tqdm import trange
import argparse

parser = argparse.ArgumentParser(description="PPO Tune Example")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster.")
args = parser.parse_args()

ray.init(address=args.ray_address, _node_ip_address=args.ray_address.split(":")[0], _redis_password=args.redis_password, log_to_driver=True)
print("Nodes:")
print(ray.nodes())

for i in trange(5, desc="Please wait, Ray resource configuration is running."):
    time.sleep(6)

#print("Cluster resources:")
#print(ray.cluster_resources())
#print("Available resources:")
#print(ray.available_resources())
#print("RT context")
#print(ray.runtime_context.get_runtime_context().get())

tune.run(
    "PPO",
    stop={"episode_reward_mean": 100, "training_iteration": 10},
    config={
        "framework": "torch",
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 32,
        "lr": tune.grid_search(np.arange (0.001, 0.02, 0.001).tolist()),
    },
    local_dir="/lus/scratch/arigazzi/ray_local/",
    sync_config=tune.SyncConfig(sync_to_driver=False),
    verbose=3,
)