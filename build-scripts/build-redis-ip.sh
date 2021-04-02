#!/bin/bash

#Install Redis
if [[ -f ./RedisIP/build/libredisip.so ]]; then
    echo "RedisIP has already been downloaded and installed"
    export REDISIP_INSTALL_PATH="$(pwd)/RedisIP/build/"
else
    if [[ ! -d "./RedisIP" ]]; then
	git clone https://github.com/Spartee/RedisIP.git
	echo "RedisIP downloaded"
    fi
    cd RedisIP
    git checkout tags/0.1.0

    echo "Building RedisIP ..."
    mkdir build
    cd build
    cmake ..
    CC=gcc CXX=g++ make
    export REDISIP_INSTALL_PATH="$(pwd)"
    echo "Finished installing RedisIP"
    cd ../../
fi
