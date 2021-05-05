#!/bin/bash

# Original file: https://github.com/NERSC/slurm-ray-cluster/blob/master/start-head.sh

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source ~/.bashrc
conda activate ;CONDA_ENV;

IP=$(hostname --ip-address)

echo "starting ray head node"
# Launch the head node
ray start --head \
          --port=;RAY_PORT; \
          --node-ip-address $IP \
          --redis-password=;REDIS_PASSWORD; \
          --num-cpus "${SLURM_CPUS_PER_TASK}" 
sleep infinity