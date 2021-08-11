#!/usr/bin/bash

#BSUB -W 00:10
#BSUB -P GEN150_SMARTSIM
#BSUB -J SmartSim
#BSUB -nnodes 1

# activate conda env if needed

# This is lsf
python launch_distributed_model_lsf.py