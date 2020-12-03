#!/bin/sh

#Install Redis
if [[ -f ./redis/src/redis-server ]]; then
    echo "Redis has already been downloaded and installed"
    export REDIS_INSTALL_PATH="$(pwd)/redis/src"
else
    if [[ ! -d "./redis" ]]; then
	git clone https://github.com/redis/redis.git redis
	cd redis
	git checkout tags/6.0.8
	cd ..
    else
	echo "Redis downloaded"
    fi
    cd redis
    echo "Downloading redis dependencies"
    CC=gcc CXX=g++ make MALLOC=libc
    export REDIS_INSTALL_PATH="$(pwd)/src"
    echo "Finished installing redis"
    cd ../
fi
