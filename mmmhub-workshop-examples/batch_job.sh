#!/bin/bash -l
#$ -A Brunel_allocation
#$ -P Gold
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:10:0
# Need one node for database and one node for proglet
#$ -pe mpi 80
# Set the name of the job.
#$ -N cpp_example

# Set the working directory to somewhere in your scratch space.
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/mmm1399/Scratch/smartsim/mmmhub-workshop-examples

# Setup the environment
module purge
source ~/tmp/smartsim/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/smartredis/install/lib
module load mpi/openmpi/3.1.5/gnu-9.2.0

# Launch the experiment
python driver_in_batch.py
