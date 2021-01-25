from smartsim import Experiment, slurm
from os import environ
import os

# Define resource variables for models,
# scripts, and orchestrator
lammps_compute_nodes = 2
lammps_ppn = 2
db_compute_nodes = 3
analysis_compute_nodes = 1

total_nodes = lammps_compute_nodes + \
              db_compute_nodes + \
              analysis_compute_nodes

# Retrieve Slurm allocation for the experiment
alloc = slurm.get_slurm_allocation(nodes=total_nodes)

# Create a SmartSim Experiment using the default
# Slurm launcher backend
experiment = Experiment("lammps_experiment")

# Define the run settings for the LAMMPS model that will
# be subsequently created.
lammps_settings = {
    "nodes": lammps_compute_nodes,
    "ntasks-per-node" : lammps_ppn,
    "executable": "lmp",
    "exe_args": "-i in.melt",
    "alloc": alloc}

# Define the run settings for the Python analysis script
# that will be subsequently created
analysis_settings = {
    "nodes": analysis_compute_nodes,
    "executable": "python",
    "exe_args": f"data_analysis.py --ranks={lammps_compute_nodes*lammps_ppn} --time=250",
    "alloc": alloc}

# Create the LAMMPS SmartSim model entity with the previously
# defined run settings
m1 = experiment.create_model("lammps_model", run_settings=lammps_settings)

# Attach the simulation input file in.melt to the entity so that
# the input file is copied into the experiment directory when it is created
m1.attach_generator_files(to_copy=["./in.melt"])

# Create the analysis SmartSim entity with the
# previously defined run settings
m2 = experiment.create_model("lammps_data_processor",run_settings=analysis_settings)

# Attach the analysis script to the SmartSim node entity so that
# the script is copied into the experiment directory when the
# experiment is generated.
m2.attach_generator_files(to_copy=["./data_analysis.py"])

# Create the SmartSim orchestrator object and database using the default
# database cluster setting of three database nodes
orc = experiment.create_orchestrator(db_nodes=db_compute_nodes,
                                     overwrite=True, alloc=alloc)

# Generate the experiment directory structure and copy the files
# attached to SmartSim entities into that folder structure.
experiment.generate(m1, m2, orc, overwrite=True)

# Start the model and orchestrator
experiment.start(m1, orc, summary=True)

# Start the data analysis script after the model is complete
experiment.start(m2, summary=True)

# When the model and node are complete, stop the
# orchestrator with the stop() call which will
# stop all running jobs when no entities are specified
#experiment.stop(orc)

# Release our system compute allocation
# experiment.release()
#slurm.release_slurm_allocation(alloc)