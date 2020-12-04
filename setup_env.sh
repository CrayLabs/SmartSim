#!/bin/bash

# set variables for RedisAI
RAI_BUILD_TYPE=${1:-"cpu"}
RAI_PT=${2:-1}
RAI_TF=${3:-1}
RAI_TFL=${4:-0}
RAI_ONNX=${5:-0}

# Ensure in Smart-Sim directory
FOUND_smartsim=$(ls | grep -c smartsim)
if [[ $FOUND_smartsim -eq 0 ]]; then
    echo "Error: Could not find smartsim package"
    echo "       Execute this script within the Smart-Sim directory."
    return 1
fi

# Update PYTHONPATH if smartsim not found already
if [[ ":$PYTHONPATH:" != *"$(pwd)"* ]]; then
    echo "Adding SmartSim to PYTHONPATH"
    export PYTHONPATH="$(pwd):${PYTHONPATH}"
else
    echo "SmartSim found in PYTHONPATH"
fi

if [[ ! -d "./third-party" ]]; then
    mkdir third-party
fi
cd third-party


# build KeyDB
source ../build-scripts/build-keydb.sh
if [ $? != 0 ]; then
    echo "ERROR: KeyDB failed to build"
    cd ..
    return 1
fi

# build redis
source ../build-scripts/build-redis.sh
if [ $? != 0 ]; then
    echo "ERROR: Redis failed to build"
    cd ..
    return 1
fi


if [[ $RAI_BUILD_TYPE == "gpu" ]]; then
    echo "Building RedisAI for GPU..."
    source ../build-scripts/build-redisai-gpu.sh $RAI_PT $RAI_TF $RAI_TFL $RAI_ONNX
    if [ $? != 0 ]; then
        echo "ERROR: RedisAI GPU failed to build"
        cd ..
        return 1
    fi
else
    echo "Building RedisAI for CPU..."
    source ../build-scripts/build-redisai-cpu.sh $RAI_PT $RAI_TF $RAI_TFL $RAI_ONNX
    if [ $? != 0 ]; then
        echo "ERROR: RedisAI GPU failed to build"
        cd ..
        return 1
    fi
fi

cd ../

# Set environment variables for SmartSim
export SMARTSIMHOME="$(pwd)"
echo "SMARTSIMHOME set to $(pwd)"
export SMARTSIM_LOG_LEVEL="info"
echo "SMARTSIM_LOG_LEVEL set to $SMARTSIM_LOG_LEVEL"
