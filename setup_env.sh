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


export ORCCONFIG="$(pwd)/smartsim/"
echo "SmartSim orchestrator config set to $(pwd)/smartsim/smartsimdb.conf"
