import ray
from ray import tune
import time
import numpy as np
import argparse
#from ray.rllib.agents import ppo
import os

parser = argparse.ArgumentParser(description="PPO Tune Example")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--redis-password", type=str, help="Password of Redis cluster.")
args = parser.parse_args()

#ray.init(address=args.ray_address, _node_ip_address=args.ray_address.split(":")[0], _redis_password=args.redis_password, log_to_driver=True)
ray.init(address=args.ray_address, _redis_password=args.redis_password)#, log_to_driver=True)#, local_mode=True)



time.sleep(5)
print("Nodes:")
print(ray.nodes())

# for i in trange(5, desc="Please wait, Ray resource configuration is running."):
#     time.sleep(6)

#print("Cluster resources:")
#print(ray.cluster_resources())
#print("Available resources:")
#print(ray.available_resources())
#print("RT context")
#print(ray.runtime_context.get_runtime_context().get())

# DEFAULT_CONFIG = ppo.PPOTrainer.merge_trainer_configs(
#     ppo.DEFAULT_CONFIG,
#     {
#         "env": "CartPole-v0",
#         # During the sampling phase, each rollout worker will collect a batch
#         # `rollout_fragment_length * num_envs_per_worker` steps in size.
#         "rollout_fragment_length": 100,
#         # Vectorize the env (should enable by default since each worker has
#         # a GPU).
#         "num_envs_per_worker": 1,
#         # During the SGD phase, workers iterate over minibatches of this size.
#         # The effective minibatch size will be:
#         # `sgd_minibatch_size * num_workers`.
#         "sgd_minibatch_size": 50,
#         # Number of SGD epochs per optimization round.
#         "num_sgd_iter": 10,
#         # Download weights between each training step. This adds a bit of
#         # overhead but allows the user to access the weights from the trainer.
#         "keep_local_weights_in_sync": True,
#         "num_gpus": 0,

# #         # *** WARNING: configs below are DDPPO overrides over PPO; you
# #         #     shouldn't need to adjust them. ***
# #         # DDPPO requires PyTorch distributed.
# #         "framework": "torch",
# #         # The communication backend for PyTorch distributed.
# #         "torch_distributed_backend": "gloo",
# #         # Learning is no longer done on the driver process, so
# #         # giving GPUs to the driver does not make sense!
# #         "num_gpus": 0,
# #         # Each rollout worker gets a GPU.
# #         "num_gpus_per_worker": 0,
# #         # Require evenly sized batches. Otherwise,
# #         # collective allreduce could fail.
# #         "truncate_episodes": True,
# #         # This is auto set based on sample batch size.
# #         "train_batch_size": -1,
#         "lr": tune.grid_search(np.arange (0.001, 0.0030, 0.0001).tolist()),
#     },
#     _allow_unknown_configs=True,
# )

tune.run(
    "PPO",
    stop={"episode_reward_mean": 100},
#     config=DEFAULT_CONFIG,
    config={
        "framework": "torch",
        "env": "CartPole-v0",
        "num_gpus": 0,
        "lr": tune.grid_search(np.arange (0.001, 0.0030, 0.0001).tolist()),
        "log_level": "DEBUG",
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
    },
    local_dir="/lus/scratch/arigazzi/ray_local/",
    #sync_config=tune.SyncConfig(sync_to_driver=False),
    verbose=3,
    fail_fast=True,
    log_to_file=True,
)

f = open("/lus/scratch/arigazzi/ray_local/a", "w")