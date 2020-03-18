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
    echo "Adding current directory to the python path"
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
        git clone https://github.com/JohnSully/KeyDB.git --branch v5.1.1 --depth=1
        cd KeyDB/ && make
        cd ..
        export PATH="$(pwd)/KeyDB/src:${PATH}"
        echo "Finished installing KeyDB"
    fi
fi

# Install Protobuf
if [[ -f ./protobuf/install/bin/protoc ]]; then
    echo "Protobuf has already been downloaded and installed"
    export PROTOBUF_INSTALL_PATH="$(pwd)/protobuf/install"
else
    if [[ ! -d "./protobuf" ]]; then
	git clone https://github.com/protocolbuffers/protobuf.git protobuf --branch master --depth=1
    else
	echo "Protobuf downloaded"
    fi
    cd protobuf
    echo "Downloading Protobuf dependencies"
    git submodule update --init --recursive
    ./autogen.sh
    ./configure --prefix="$(pwd)/install"
    make
    make check
    make install
    export PROTOBUF_INSTALL_PATH="$(pwd)/install"
    echo "Finished installing Protobuf"
    cd ../
fi

# Install Hiredis
if ls ./hiredis/install/lib/libhiredis* 1>/dev/null 2>&1; then
    echo "Hiredis has already been downloaded and installed"
    export HIREDIS_INSTALL_PATH="$(pwd)/hiredis/install"
else
    if [[ ! -d "./hiredis" ]]; then
	git clone https://github.com/redis/hiredis.git hiredis --branch master --depth=1
	echo "Hiredis downloaded"
    fi
    cd hiredis
    make PREFIX="$(pwd)/install"
    make PREFIX="$(pwd)/install" install
    cd ../
    export HIREDIS_INSTALL_PATH="$(pwd)/hiredis/install"
    echo "Finished installing Hiredis"
fi

#Install Redis-plus-plus
if ls ./redis-plus-plus/install/lib/libredis++* 1>/dev/null 2>&1; then
    echo "Redis-plus-plus has already been downloaded and installed"
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
else
    if [[ ! -d "./redis-plus-plus" ]]; then
        git clone https://github.com/sewenew/redis-plus-plus.git redis-plus-plus --branch master --depth=1
        echo "Redis-plus-plus downloaded"
    fi
    cd redis-plus-plus
    ex -s -c '2i|SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)' -c x CMakeLists.txt
    mkdir compile
    cd compile
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HIREDIS_INSTALL_PATH}" -DCMAKE_INSTALL_PREFIX="$(pwd)/../install" ..
    make
    make install
    cd ../../
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
    echo "Finished installing Redis-plus-plus"
fi

cd ../

export ORCCONFIG="$(pwd)/smartsim/"
echo "SmartSim orchestrator config set to $(pwd)/smartsim/smartsimdb.conf"
