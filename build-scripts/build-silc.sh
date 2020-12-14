#!/bin/bash

echo "Retrieving SILC library"
git clone https://github.com/Spartee/SILC.git silc
cd silc
git checkout develop

echo "Making SILC dependencies"
make deps
source setup_env.sh
echo "Making SILC python client"
make pyclient