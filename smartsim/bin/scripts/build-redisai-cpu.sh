#!/bin/bash

#Install Redis
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Detect OS type
if [[ "$OSTYPE" == "linux"* ]]; then
        OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="mac"
else
        OS="unknown"
fi

#Install RedisAI
if [[ -f "$DIR/../../lib/redisai.so" ]]; then
    echo "RedisAI installed"
else
    if [[ ! -d "$DIR/../../.third-party/RedisAI" ]]; then
        git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.2 --depth=1 $DIR/../../.third-party/RedisAI
        echo "RedisAI downloaded"
    fi
    echo "Downloading RedisAI CPU ML Runtimes"
    CC=gcc CXX=g++ WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 bash $DIR/../../.third-party/RedisAI/get_deps.sh cpu

    echo "Building RedisAI CPU ML Runtimes"
    CC=gcc CXX=g++ GPU=0 WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 WITH_UNIT_TESTS=0 make -C $DIR/../../.third-party/RedisAI/opt clean build

    if [ -f "$DIR/../../.third-party/RedisAI/install-cpu/redisai.so" ]; then
        cp $DIR/../../.third-party/RedisAI/install-cpu/redisai.so $DIR/../../lib/
        cp -r $DIR/../../.third-party/RedisAI/install-cpu/backends $DIR/../../lib/
        rm -rf $DIR/../../.third-party
        echo "Finished installing RedisAI"
    else
        echo "ERROR: RedisAI failed to build"
        return 1
    fi
fi
