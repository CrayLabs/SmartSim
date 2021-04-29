#!/bin/bash

#Install Redis
if [[ -f ./redis/src/redis-server ]]; then
    echo "Redis has already been downloaded and installed"
    export REDIS_INSTALL_PATH="$(pwd)/redis/src"
else
    if [[ ! -d "./redis" ]]; then
	git clone https://github.com/redis/redis.git --branch 6.0.8 --depth 1 redis
    else
	echo "Redis downloaded"
    fi
    cd redis
    echo "Building Redis"
    CC=gcc CXX=g++ make MALLOC=libc
    if [[ -f ./src/redis-server ]]; then
        if [[ -f ../../smartsim/bin/redis-* ]]; then
            rm ../../smartsim/bin/redis-*
        fi
        cp ./src/redis-server ./src/redis-cli ../../smartsim/bin
        export REDIS_INSTALL_PATH="$(pwd)/src"
        echo "Finished installing redis"
    fi
    cd ../
fi
