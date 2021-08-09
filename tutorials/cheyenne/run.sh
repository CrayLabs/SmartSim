#!/bin/bash

#PBS -l select=3:ncpus=20:mpiprocs=20
#PBS -l walltime=00:10:00
#PBS -A NCGD0048
#PBS -q economy
#PBS -N SmartSim

# activate conda env if needed
python launch_distributed_model.py