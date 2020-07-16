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

# Install Protobuf
if [[ -f ./protobuf/install/bin/protoc ]]; then
    echo "Protobuf has already been downloaded and installed"
    export PROTOBUF_INSTALL_PATH="$(pwd)/protobuf/install"
    export LD_LIBRARY_PATH="$PROTOBUF_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    export PATH="$PROTOBUF_INSTALL_PATH/bin":$PATH
else
    if [[ ! -d "./protobuf" ]]; then
	git clone https://github.com/protocolbuffers/protobuf.git protobuf
	cd protobuf
	git checkout tags/v3.11.3
	cd ..
    else
	echo "Protobuf downloaded"
    fi
    cd protobuf
    echo "Downloading Protobuf dependencies"
    git submodule update --init --recursive
    ./autogen.sh
    ./configure --prefix="$(pwd)/install"
    make -j 8
    make check -j 8
    make install
    export PROTOBUF_INSTALL_PATH="$(pwd)/install"
    export LD_LIBRARY_PATH="$PROTOBUF_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    export PATH="$PROTOBUF_INSTALL_PATH/bin":$PATH
    echo "Finished installing Protobuf"
    cd ../
fi

# Install Hiredis
if ls ./hiredis/install/lib/libhiredis* 1>/dev/null 2>&1; then
    echo "Hiredis has already been downloaded and installed"
    export HIREDIS_INSTALL_PATH="$(pwd)/hiredis/install"
    export LD_LIBRARY_PATH="$HIREDIS_INSTALL_PATH/lib":$LD_LIBRARY_PATH
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
    export LD_LIBRARY_PATH="$HIREDIS_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Finished installing Hiredis"
fi

#Install Redis-plus-plus
if ls ./redis-plus-plus/install/lib/libredis++* 1>/dev/null 2>&1; then
    echo "Redis-plus-plus has already been downloaded and installed"
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
    export LD_LIBRARY_PATH="$REDISPP_INSTALL_PATH/lib":$LD_LIBRARY_PATH
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
    make -j 2
    make install
    cd ../../
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
    export LD_LIBRARY_PATH="$REDISPP_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Finished installing Redis-plus-plus"
fi

cd ../

# Set environment variables for SmartSim
export SMARTSIMHOME="$(pwd)"
echo "SMARTSIMHOME set to $(pwd)"
export SMARTSIM_LOG_LEVEL="info"
echo "SMARTSIM_LOG_LEVEL set to $SMARTSIM_LOG_LEVEL"
