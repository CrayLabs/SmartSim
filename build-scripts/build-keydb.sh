#!/bin/bash

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
