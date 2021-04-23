#!/bin/bash
source ~/.bashrc
conda activate ;CONDA_ENV;
IP=$(ip -oneline -family inet addr list ipogif0 | head --lines 1 | grep --perl-regexp --only-matching 'inet \K[\d.]+')
#echo $IP:6380 > $HOME/ray_head_node
ray start --head --port=;RAY_PORT;\
          --redis-password=;REDIS_PASSWORD; \
          --node-ip-address=$IP
          
sleep infinity