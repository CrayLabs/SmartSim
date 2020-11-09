from smartsim import Experiment

# Create the Experiment object
experiment = Experiment("lammps_crack", launcher="local")

# Set the run settings for each member of the
# ensemble.
run_settings = {
    "executable": "mpirun",
    "exe_args": "-np 2 lmp_mpi -i in.crack"
}

# Set the parameter space for the ensemble
# The default strategy is to generate all permutations
# so all permutations of STEPS and THERMO will
# be generated as a single model
model_params = {
    "STEPS": [20000, 40000],
    "THERMO": [150, 250]
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
experiment.start(ensemble, summary=True)

# show what happened in the experiment
print(experiment.summary())
