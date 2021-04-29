#!/bin/bash

#Install RedisAI
if [[ -f ./RedisAI/install-cpu/redisai.so ]]; then
    echo "RedisAI CPU has already been downloaded and installed"
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/RedisAI/install-cpu"
else
    if [[ ! -d "./RedisAI" ]]; then
        git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.2 --depth=1 RedisAI
    else
        echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI CPU dependencies"
    CC=gcc CXX=g++ VERBOSE=1 WITH_ORT=0 bash get_deps.sh cpu
    CC=gcc CXX=g++ GPU=0 WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 SHOW=1 make -C opt clean build
    if [ -f "./install-cpu/redisai.so" ]; then
        if [[ -f ../../smartsim/lib/redisai.so ]]; then
            rm ../../smartsim/lib/redisai.so
        fi
        cp  ./install-cpu/redisai.so ../../smartsim/lib/
        export REDISAI_CPU_INSTALL_PATH="$(pwd)/install-cpu"
        echo "Finished installing RedisAI"
        cd ../
    else
        echo "ERROR: RedisAI failed to build"
        return 1
    fi
fi
