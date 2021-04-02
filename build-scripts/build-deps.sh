#!/bin/bash

# set variables for RedisAI
RAI_BUILD_TYPE=${1:-"cpu"}
RAI_PT=${2:-1}
RAI_TF=${3:-1}
RAI_TFL=${4:-0}
RAI_ONNX=${5:-0}


# make deps directory
if [[ ! -d "./third-party" ]]; then
    mkdir third-party
fi
cd third-party

# build redis
../build-scripts/build-redis.sh
if [ $? != 0 ]; then
    echo "ERROR: Redis failed to build"
    cd ..
    exit 1
fi

# Build RedisAI
if [[ $RAI_BUILD_TYPE == "gpu" ]]; then
    echo "Building RedisAI for GPU..."
    source ../build-scripts/build-redisai-gpu.sh $RAI_PT $RAI_TF $RAI_TFL $RAI_ONNX
    if [ $? != 0 ]; then
        echo "ERROR: RedisAI GPU failed to build"
        cd ..
        exit 1
    fi
else
    echo "Building RedisAI for CPU..."
    source ../build-scripts/build-redisai-cpu.sh $RAI_PT $RAI_TF $RAI_TFL $RAI_ONNX
    if [ $? != 0 ]; then
        echo "ERROR: RedisAI GPU failed to build"
        cd ..
        exit 1
    fi
fi

# build RedisIP
../build-scripts/build-redis-ip.sh
if [ $? != 0 ]; then
    echo "ERROR: RedisIP failed to build"
    cd ..
    exit 1
fi


cd ../