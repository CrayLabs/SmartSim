#!/bin/bash

# Original file: https://github.com/NERSC/slurm-ray-cluster/master/start-worker.sh

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source ~/.bashrc
conda activate ;CONDA_ENV;

echo "starting ray worker node"
ray start --address ;HEAD_ADDRESS; \
          --redis-password=;REDIS_PASSWORD; \
          --num-cpus "${SLURM_CPUS_PER_TASK}"
          
sleep infinity