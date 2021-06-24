#!/bin/bash
module purge
# Fill out the required 
module load openmpi

load_conda
conda activate smartsim-test
launch_database_cluster.py
