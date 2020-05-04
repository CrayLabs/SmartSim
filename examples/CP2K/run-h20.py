from smartsim import Experiment

# Initialize the experiment using the default launcher "slurm"
experiment = Experiment("h2o", launcher="slurm")

# Get an allocation through Slurm to launch entities
# extra arguments can be specified through kwargs
# e.g. "qos"="interactive"
alloc = experiment.get_allocation(nodes=4, partition="gpu")

# List of parameters that will form the ensemble
# one ensemble member will be created per value within
# the STEPS array
ensemble_params = {"STEPS": [10, 15, 20, 25]}

# Define how and where the CP2K models should be
# executed. Allocation is provided with the "alloc" keyword
run_settings = {"executable": "cp2k.psmp",
                "partition": "gpu",
                "exe_args": "-i h2o.inp",
                "nodes": 1,
                "alloc": alloc}

# Create the ensemble with settings listed above
experiment.create_ensemble("h2o-1", params=ensemble_params, run_settings=run_settings)

# Generate an experiment directory and create the
# ensembles based on the settings supplied to the
# ensemble, which in this case will create 4 models
# each with a different value in their configuration
# file for number of steps
experiment.generate(model_files="./h2o.inp")

# launch the ensemble. Since we dont specify which
# ensemble to run, launch all entities defined within
# the experiment.
experiment.start()

# Since Experiment.start() is non-blocking when using
# the Slurm launcher, poll slurm for
experiment.poll()

# release all allocations obtained by this experiment
experiment.release()


