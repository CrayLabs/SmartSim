#!/bin/bash

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
    echo $PYTHONPATH
else
    echo "SmartSim found in PYTHONPATH"
fi

if [[ -d "./silc" ]]; then
    cd ./silc
    source setup_env.sh
    cd ..
fi

# Set environment variables for SmartSim
export SMARTSIMHOME="$(pwd)"
echo "SMARTSIMHOME set to $(pwd)"
export SMARTSIM_LOG_LEVEL="info"
echo "SMARTSIM_LOG_LEVEL set to $SMARTSIM_LOG_LEVEL"
