#!/bin/bash

#
# Basically just building the documentation and kicking of the deployment.
#

DESTINATION="build"
PKG="smartsim"

# Help output
print_help() {
    script_name=$(basename "$0")
    # note: would this be cleaner with multi-line strings?
    echo "${script_name}: deploys built ${PKG} python module into ${DESTINATION}"
    echo
    echo "Usage: ${script_name} [-h]"
    echo
    echo "Requirements:"
    echo "  - Some subset of ${PKG} is built."
}

# Print the help output if an error occurred
finalize() {
    exitcode=$?
    if [ ${exitcode} -ne 0 ]; then
        print_help
    else
        echo "${PKG} python module deployed into: ${DESTINATION}"
    fi

    exit $exitcode
}

trap finalize EXIT

#
# Parse args
#

case "$1" in
    -h|--help)
    print_help
    ;;
esac

#
# Setup
#

# Wipe previous ${DESTINATION}
# Move to tmp for extra level of security to prevent wiping our home directory
if [ -d ${DESTINATION} ]; then
  mv ${DESTINATION} tmp
  rm -rf tmp
fi

# Only copy if built binaries are found
if compgen -G "dist/*.tar.gz" > /dev/null; then
    mkdir -p ${DESTINATION}
    cp dist/*.tar.gz ${DESTINATION}
else
    echo "Warning: No built file found in dist"
fi


#
# Docs
#

if [ -d _build ]; then
    cp -r _build ${DESTINATION}/_build
else
    echo "Warning: No build docs found"
fi
