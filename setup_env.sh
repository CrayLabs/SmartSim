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
	git checkout tags/v6.0.8
	cd ..
    else
	echo "Redis downloaded"
    fi
    cd redis
    echo "Downloading redis dependencies"
    CC=gcc CXX=g++ make
    export REDIS_INSTALL_PATH="$(pwd)/src"
    echo "Finished installing redis"
    cd ../
fi

#Install RedisAI CPU
if [[ -f ./RedisAI/install-cpu/redisai.so ]]; then
    echo "RedisAI CPU has already been downloaded and installed"
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/RedisAI/install-cpu"
else
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
    CC=gcc CXX=g++ bash get_deps.sh cpu
    CC=gcc CXX=g++ ALL=1 make -C opt clean build
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/install-cpu"
    echo "Finished installing RedisAI"
    cd ../
fi

cd ../

# Set environment variables for SmartSim
export SMARTSIMHOME="$(pwd)"
echo "SMARTSIMHOME set to $(pwd)"
export SMARTSIM_LOG_LEVEL="info"
echo "SMARTSIM_LOG_LEVEL set to $SMARTSIM_LOG_LEVEL"
