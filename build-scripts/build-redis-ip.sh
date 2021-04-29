#!/bin/bash

#Install Redis
if [[ -f ./RedisIP/build/libredisip.so ]]; then
    echo "RedisIP has already been downloaded and installed"
    export REDISIP_INSTALL_PATH="$(pwd)/RedisIP/build/"
else
    if [[ ! -d "./RedisIP" ]]; then
	git clone https://github.com/Spartee/RedisIP.git --branch master --depth 1
	echo "RedisIP downloaded"
    fi
    cd RedisIP
    echo "Building RedisIP ..."
    mkdir build
    cd build
    cmake ..
    CC=gcc CXX=g++ make
    if [[ -f ./libredisip.so ]]; then
        if [[ -f ../../../smartsim/lib/libredisip.so ]]; then
            rm ../../../smartsim/lib/libredisip.so
        fi
        cp ./libredisip.so ../../../smartsim/lib
        export REDISIP_INSTALL_PATH="$(pwd)"
        echo "Finished installing RedisIP"
    fi
    cd ../../
fi
