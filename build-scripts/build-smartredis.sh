#!/bin/bash

echo "Retrieving SmartRedis library"
git clone https://github.com/CrayLabs/SmartRedis.git smartredis
cd smartredis
git checkout develop

echo "Making smartredis dependencies"
make deps
source setup_env.sh
echo "Making smartredis python client"
make pyclient