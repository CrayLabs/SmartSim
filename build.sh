#!/bin/bash

# make sure our bash_profile gets loaded
source ~/.bash_profile

# Build/Install Dependencies
source setup_env.sh
pip install -r requirements.txt

# Build library
python setup.py sdist

./deploy.sh
