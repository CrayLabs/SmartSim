#!/bin/bash

echo "Retrieving smartredis library"
git clone https://github.com/Spartee/smartredis.git smartredis
cd smartredis
git checkout develop

echo "Making smartredis dependencies"
make deps
source setup_env.sh
echo "Making smartredis python client"
make pyclient