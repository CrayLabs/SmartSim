#!/bin/bash

#Install Redis
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Detect OS type
if [[ "$OSTYPE" == "linux"* ]]; then
        OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "ERROR: RedisAI GPU does not support MacOS"
else
        OS="unknown"
fi

#Install RedisAI
if [[ -f "$DIR/../../lib/redisai.so" ]]; then
    echo "RedisAI installed"
else
    # check for cudnn includes
    if [ -z "$CUDNN_INCLUDE_DIR" ]; then
        echo "WARNING: CUDNN_INCLUDE_DIR is not set!!!"
    fi

    # check for cudnn library
    if [ -z "$CUDNN_LIBRARY" ]; then
        echo "WARNING: CUDNN_LIBRARY is not set!!!"
    fi

    if [[ ! -d "$DIR/../../.third-party/RedisAI" ]]; then
        git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.2 --depth=1 $DIR/../../.third-party/RedisAI
        echo "RedisAI downloaded"
    fi
    echo "Downloading RedisAI GPU ML Runtimes"
    CC=gcc CXX=g++ WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 bash $DIR/../../.third-party/RedisAI/get_deps.sh gpu

    echo "Building RedisAI GPU ML Runtimes"
    CC=gcc CXX=g++ GPU=1 WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 WITH_UNIT_TESTS=0 make -C $DIR/../../.third-party/RedisAI/opt clean build

    if [ -f "$DIR/../../.third-party/RedisAI/install-gpu/redisai.so" ]; then
        cp $DIR/../../.third-party/RedisAI/install-gpu/redisai.so $DIR/../../lib/
        cp -r $DIR/../../.third-party/RedisAI/install-gpu/backends $DIR/../../lib/
        rm -rf $DIR/../../.third-party
        echo "Finished installing RedisAI"
    else
        echo "ERROR: RedisAI failed to build"
        return 1
    fi
fi
