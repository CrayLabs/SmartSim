#!/bin/bash

#Install Redis
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Detect OS type
if [[ "$OSTYPE" == "linux"* ]]; then
        OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "ERROR: RedisAI GPU does not support MacOS"
        return 1
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
        GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.3 --depth=1 $DIR/../../.third-party/RedisAI
	    cp $DIR/../modules/FindTensorFlow.cmake $DIR/../../.third-party/RedisAI/opt/cmake/modules/ 
        echo "RedisAI downloaded"
    fi
    echo "Downloading RedisAI GPU ML Runtimes"
    CC=gcc CXX=g++ WITH_PT=0 WITH_TF=$2 WITH_TFLITE=0 WITH_ORT=$4 bash $DIR/../../.third-party/RedisAI/get_deps.sh gpu

    echo "Building RedisAI GPU ML Runtimes"
    CC=gcc CXX=g++ GPU=1 WITH_PT=$1 WITH_TF=$2 WITH_TFLITE=$3 WITH_ORT=$4 WITH_UNIT_TESTS=0 make -j $NPROC -C $DIR/../../.third-party/RedisAI/opt clean build

    if [ -f "$DIR/../../.third-party/RedisAI/install-gpu/redisai.so" ]; then
        cp $DIR/../../.third-party/RedisAI/install-gpu/redisai.so $DIR/../../lib/
        cp -r $DIR/../../.third-party/RedisAI/install-gpu/backends $DIR/../../lib/
        rm -rf $DIR/../../.third-party
        # remove the need for the user to set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH (mac)
        # by copying shared libs to lib
        if [[ $1 == "1" ]]; then
            mkdir -p $DIR/../../lib/backends/redisai_torch/lib/
            cp $SITE_PACKAGES/torch/lib/* $DIR/../../lib/backends/redisai_torch/lib/
        fi
        echo "Finished installing RedisAI"
    else
        echo "ERROR: RedisAI failed to build"
        return 1
    fi
fi
