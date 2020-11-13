from smartsim import Experiment
from smartsim import slurm

# intialize our Experiment and obtain
# allocations for our ensemble suite
experiment = Experiment("double_gyre", launcher="slurm")

iv24_opts = {
    "partition": "iv24",
    "ntasks-per-node": 48
}
iv24_alloc = slurm.get_slurm_allocation(nodes=16, add_opts=iv24_opts)

knl_opts = {
    "partition": "knl",
    "ntasks-per-node": 48
}
knl_alloc = slurm.get_slurm_allocation(nodes=8, add_opts=knl_opts)

high_res_model_params = {
    "KH": [250, 500, 750, 1000],
    "KHTH": [250, 500],
    "x_resolution": 80,
    "y_resolution": 40,
    "months": 3
}

iv24_run_settings = {
    "nodes": 2,
    "ntasks-per-node": 48,
    "executable": "MOM6",
    "alloc": iv24_alloc
}

knl_run_settings = {
    "nodes": 1,
    "ntasks-per-node": 96,
    "executable": "MOM6",
    "alloc": knl_alloc
}

# create the ensemble that will run on the iv24 nodes
high_res_iv24 = experiment.create_ensemble(
    "high-res-iv24",
    params=high_res_model_params,
    run_settings=iv24_run_settings
    )
high_res_iv24.attach_generator_files(
    to_copy=["./MOM6_base_config"],
    to_configure=["./MOM6_base_config/input.nml",
                  "./MOM6_base_config/MOM_input"]
    )

# create the ensemble that will run on the knights landing nodes
high_res_knl = experiment.create_ensemble(
    "high-res-knl",
    params=high_res_model_params,
    run_settings=knl_run_settings
    )
high_res_knl.attach_generator_files(
    to_copy=["./MOM6_base_config"],
    to_configure=["./MOM6_base_config/input.nml",
                  "./MOM6_base_config/MOM_input"]
    )

# configure and create the low resolution
# double gyre ensemble
low_res_model_params = {
    "KH": [250, 500, 750, 1000],
    "KHTH": [250, 500, 750, 1000],
    "x_resolution": 40,
    "y_resolution": 40,
    "months": 3
}
low_res_run_settings = {
    "nodes": 1,
    "ntasks-per-node": 48,
    "executable": "MOM6",
    "alloc": iv24_alloc
}

low_res_iv24 = experiment.create_ensemble(
    "low-res-iv24",
    params=low_res_model_params,
    run_settings=low_res_run_settings
    )
low_res_iv24.attach_generator_files(
    to_copy=["./MOM6_base_config"],
    to_configure=["./MOM6_base_config/input.nml",
                  "./MOM6_base_config/MOM_input"])

# generate the files needed for all of our models
experiment.generate(high_res_knl, high_res_iv24, low_res_iv24)

# start the two high resolution models on the IV24 and KNL
# partitions.
experiment.start(high_res_knl, high_res_iv24, block=True, summary=True)

# print out the statuses of the model we just ran
iv24_statuses = experiment.get_status(high_res_iv24)
print(f"Statuses of IV24 Models: {iv24_statuses}")

knl_statuses = experiment.get_status(high_res_knl)
print(f"Statuses of KNL Models: {knl_statuses}")

# Release the KNL partition because we dont need it anymore
slurm.release_slurm_allocation(knl_alloc)

# start the low resolution simulation on the same
# allocation as the IV24 high resolution model
experiment.start(low_res_iv24, block=True, summary=False)

# print the statuses of the low resolution ensemble
# after it has completed.
iv24_low_res_statuses = experiment.get_status(low_res_iv24)
print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

# Release the iv24 partition
slurm.release_slurm_allocation(iv24_alloc)