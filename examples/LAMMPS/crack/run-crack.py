from smartsim import Experiment


# Create the Experiment object
experiment = Experiment("lammps_crack", launcher="slurm")

# get an 8 node allocation with 48 processors per node
# in exclusive mode
alloc = experiment.get_allocation(nodes=8, ppn=48, exclusive=None)

# Set the run settings for each member of the
# ensemble. This includes the allocation id
# that we just obtained.
run_settings = {
    "executable": "lmp",
    "exe_args": "-i in.crack",
    "nodes": 1,
    "ppn": 48,
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
    "STEPS": [10000, 20000],
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
experiment.generate()

# Start the experiment
experiment.start()

# Poll the models as they run
experiment.poll()

# release the allocation obtained for this experiment
experiment.release()
