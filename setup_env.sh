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
    echo "New PYTHONPATH:"
    echo $PYTHONPATH
else
    echo "$(pwd) already found in PYTHONPATH:"
    echo "$PYTHONPATH"
fi

# Setup the Smart Simulation library environment variables
export SMARTSIMHOME="$(pwd)/examples/"
echo "SmartSim Home set to: $(pwd)/examples/"
