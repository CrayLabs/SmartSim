from smartsim import Experiment
from smartsim import slurm

# Create the Experiment object
experiment = Experiment("lammps_crack", launcher="slurm")

# get an 8 node allocation with 48 processors per node
# in exclusive mode
add_opts = {
    "exclusive": None,
    "ntasks-per-node": 48
}
alloc = slurm.get_slurm_allocation(nodes=8, add_opts=add_opts)

# Set the run settings for each member of the
# ensemble. This includes the allocation id
# that we just obtained.
run_settings = {
    "executable": "lmp",
    "exe_args": "-i in.crack",
    "nodes": 1,
    "ntasks-per-node": 48,
    "env_vars": {
        "OMP_NUM_THREADS": 1
    },
    "alloc": alloc
}

# Set the parameter space for the ensemble
# The default strategy is to generate all permuatations
# so all permutations of STEPS and THERMO will
# be generated as a single model
model_params = {
    "STEPS": [100000, 200000],
    "THERMO": [150, 200, 250, 300]
}

# Create ensemble with the model params and
# run settings defined
ensemble = experiment.create_ensemble("crack",
                                      params=model_params,
                                      run_settings=run_settings)

# attach files to be generated at runtime
# in each model directory where the executable
# will be invoked
ensemble.attach_generator_files(to_configure="./in.crack")
experiment.generate(ensemble, overwrite=True)

# Start the experiment
# block still all models are finished
# print a pre-launch summary
experiment.start(ensemble, block=True, summary=True)

# release the allocation obtained for this experiment
slurm.release_slurm_allocation(alloc)

# print out the summary of what the experiment ran
print(experiment.summary())