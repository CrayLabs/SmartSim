#!/bin/bash

#Install Redis
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")

#Install Redis
if [[ -f "$DIR/../redis-server" ]]; then
    echo "Redis installed"
else
    if [[ ! -d "$DIR/../../.third-party/redis" ]]; then
	git clone https://github.com/redis/redis.git --branch 6.0.8 --depth 1 $DIR/../../.third-party/redis
    else
	echo "Redis downloaded"
    fi
    echo "Building Redis"
    CC=gcc CXX=g++ make -j $NPROC -C $DIR/../../.third-party/redis MALLOC=libc
    cp $DIR/../../.third-party/redis/src/redis-server $DIR/../../.third-party/redis/src/redis-cli $DIR/../../bin
    echo "Finished installing redis"
fi
