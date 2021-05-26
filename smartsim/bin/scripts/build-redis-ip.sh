#!/bin/bash

#Install RedisIP
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# is already installed?
if [[ -f "$DIR/../../lib/libredisip.so" ]]; then
    echo "RedisIP installed"
else
    if [[ ! -d "$DIR/../../.third-party/RedisIP" ]]; then
        git clone https://github.com/Spartee/RedisIP.git --branch master --depth 1 $DIR/../../.third-party/RedisIP
        echo "RedisIP downloaded"
    fi
    echo "Building RedisIP ..."
    mkdir $DIR/../../.third-party/RedisIP/build
    cmake -B $DIR/../../.third-party/RedisIP/build/ -S $DIR/../../.third-party/RedisIP/
    CC=gcc CXX=g++ make -C $DIR/../../.third-party/RedisIP/build/
    cp $DIR/../../.third-party/RedisIP/build/libredisip.so $DIR/../../lib/
    chmod u+x $DIR/../../lib/libredisip.so
    echo "Finished installing RedisIP"
fi

