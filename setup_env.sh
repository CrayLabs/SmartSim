#!/bin/sh

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

# Install Keydb
found_keydb=$(which keydb-server > /dev/null 2<&1)
if [[ -x "$found_keydb" ]] ; then
    echo "KeyDB is installed"
else
    if [[ -d "./KeyDB" ]] ; then
        echo "KeyDB has already been downloaded"
        export PATH="$(pwd)/KeyDB/src:${PATH}"
        echo "Added KeyDB to PATH"
    else
        echo "Installing KeyDB"
        git clone https://github.com/JohnSully/KeyDB.git --branch v5.3.3 --depth=1
        cd KeyDB/
	CC=gcc CXX=g++ make -j 2
        cd ..
        export PATH="$(pwd)/KeyDB/src:${PATH}"
        echo "Finished installing KeyDB"
    fi
fi

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



#Install RedisAI
if [[ -f ./RedisAI/install-gpu/redisai.so ]]; then
    echo "RedisAI GPU has already been downloaded and installed"
    export REDISAI_GPU_INSTALL_PATH="$(pwd)/RedisAI/install-gpu"
else
    if [ -z "$CUDNN_INCLUDE_DIR" ]
    then
        echo "\$CUDNN_INCLUDE_DIR is not set, dependencies will likely fail to build"
    fi

    if [ -z "$CUDNN_LIBRARY" ]
    then
        echo "\$CUDNN_LIBRARY is not set, dependencies will likely fail to build"
    fi

    if [[ ! -d "./RedisAI" ]]; then
	git clone https://github.com/RedisAI/RedisAI.git RedisAI
	cd RedisAI
	git checkout tags/v1.0.2
	cd ..
    else
	echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI dependencies"
    CC=gcc CXX=g++ bash get_deps.sh gpu
    # TODO: enable TF and ONNX builds.
    CC=gcc CXX=g++ ALL=1 make -C opt clean build GPU=1 WITH_TF=0 WITH_TFLITE=0 WITH_ORT=0
    export REDISAI_GPU_INSTALL_PATH="$(pwd)/install-gpu"
    echo "Finished installing RedisAI"
    cd ../
fi

cd ../

# Set environment variables for SmartSim
export SMARTSIMHOME="$(pwd)"
echo "SMARTSIMHOME set to $(pwd)"
export SMARTSIM_LOG_LEVEL="info"
echo "SMARTSIM_LOG_LEVEL set to $SMARTSIM_LOG_LEVEL"
