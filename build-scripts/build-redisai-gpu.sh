#!/bin/bash

#Install RedisAI
if [[ -f ./RedisAI/install-gpu/redisai.so ]]; then
    echo "RedisAI GPU has already been downloaded and installed"
    export REDISAI_GPU_INSTALL_PATH="$(pwd)/RedisAI/install-gpu"
else

    # check for cudnn includes
    if [ -z "$CUDNN_INCLUDE_DIR" ]; then
        echo "ERROR: CUDNN_INCLUDE_DIR is not set"
        return 1
    else
        echo "Found CUDNN_INCLUDE_DIR: $CUDNN_INCLUDE_DIR "
        if [ -f "$CUDNN_INCLUDE_DIR/cudnn.h" ]; then
            echo "Found cudnn.h at $CUDNN_INCLUDE_DIR"
        else
            echo "ERROR: could not find cudnn.h at $CUDNN_INCLUDE_DIR"
            return 1
        fi
    fi

    # check for cudnn library
    if [ -z "$CUDNN_LIBRARY" ]; then
        echo "ERROR: CUDNN_LIBRARY is not set"
        return 1
    else
        echo "Found CUDNN_LIBRARY: $CUDNN_LIBRARY"
        if [ -f "$CUDNN_LIBRARY/libcudnn.so" ]; then
            echo "Found libcudnn.so at $CUDNN_LIBRARY"
        else
            echo "ERROR: could not find libcudnn.so at $CUDNN_LIBRARY"
            return 1
        fi
    fi


    if [[ ! -d "./RedisAI" ]]; then
        git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.2 --depth=1 RedisAI
    else
        echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI GPU dependencies"
    CC=gcc CXX=g++ VERBOSE=1 WITH_ORT=0 bash get_deps.sh gpu
    # TODO: enable TF and ONNX builds.
    CC=gcc CXX=g++ GPU=1 WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 SHOW=1 make -C opt clean build

    if [ -f "./install-gpu/redisai.so" ]; then
        if [[ -f ../../smartsim/lib/redisai.so ]]; then
            rm ../../smartsim/lib/redisai.so
        fi
        cp  ./install-gpu/redisai.so ../../smartsim/lib/
        export REDISAI_GPU_INSTALL_PATH="$(pwd)/install-gpu"
        echo "Finished installing RedisAI"
        cd ../
    else
        echo "ERROR: RedisAI failed to build"
        return 1
    fi
fi
