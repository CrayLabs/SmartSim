from smartsim import Experiment
import os

# Define resource variables that we will
# use to get and manage system resources
lammps_compute_nodes = 2
db_compute_nodes = 3
analysis_compute_nodes = 1

total_compute_nodes = lammps_compute_nodes  + db_compute_nodes + analysis_compute_nodes

# Define the a variable for the max number of processes
# per compute node that we expect to need
ppn = 2

# Create a SmartSim Experiment using the default
# Slurm launcher backend
experiment = Experiment("lammps_experiment")

# Fetch a compute resource allocation using SmartSim
# Experiment API
alloc = experiment.get_allocation(total_compute_nodes, ppn=ppn)


# Define the run settings for the LAMMPS model that will
# be subsequently created.
lammps_settings = {
    "nodes": lammps_compute_nodes,
    "ppn" : ppn,
    "executable": "lmp",
    "exe_args": "-i in.melt",
    "alloc": alloc}

# Define the run settings for the Python analysis script
# that will be subsequently created
analysis_settings = {
    "nodes": analysis_compute_nodes,
    "executable": "python smartsim_node.py",
    "exe_args": f"--ranks={lammps_compute_nodes*ppn} --time=250",
    "alloc": alloc}

# Create the LAMMPS SmartSim model entity with the previously
# defined run settings
m1 = experiment.create_model("lammps_melt_model", run_settings=lammps_settings)

# Attach the simulation input file in.melt to the entity so that
# the input file is copied into the experiment directory when it is created
m1.attach_generator_files(to_copy=[f"{os.getcwd()}/in.melt"])

# Create the analysis SmartSim node entity with the
# previously defined run settings
n1 = experiment.create_node("lammps_data_processor",run_settings=analysis_settings)

# Attach the analysis script to the SmartSim node entity so that
# the script is copied into the experiment directory when the
# experiment is generated.
n1.attach_generator_files(to_copy=[f"{os.getcwd()}/smartsim_node.py"])

# Create the SmartSim orchestrator object and database using the default
# database cluster setting of three database nodes
orc = experiment.create_orchestrator_cluster(alloc, overwrite=True)

# Generate the experiment directory structure and copy the files
# attached to SmartSim entities into that folder structure.
experiment.generate()

# Start the experiment
experiment.start()

# Poll the status of the SmartSim model and node in a blocking
# manner until both are completed
experiment.poll()

# When the model and node are complete, stop the
# orchestrator with the stop() call which will
# stop all running jobs when no entities are specified
experiment.stop()

# Release our system compute allocation
experiment.release()
