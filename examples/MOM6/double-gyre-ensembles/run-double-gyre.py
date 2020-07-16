from smartsim import Experiment
from smartsim.generation import Generator

# intialize our Experiment and obtain
# an allocation for our ensemble suite
experiment = Experiment("double_gyre")
iv24_alloc = experiment.get_allocation(nodes=16, ppn=48,
                                       partition="iv24", exclusive=None)
knl_alloc = experiment.get_allocation(nodes=8, ppn=96,
                                      partition="knl", exclusive=None)


high_res_model_params = {
    "KH": [250, 500, 750, 1000],
    "KHTH": [250, 500],
    "x_resolution": 80,
    "y_resolution": 40,
    "months": 3
}

iv24_run_settings = {
    "nodes": 2,
    "ppn": 48,
    "executable": "MOM6",
    "alloc": iv24_alloc
}

knl_run_settings = {
    "nodes": 1,
    "ppn": 96,
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

# intialize a Generator instance for greater control
# over when and where generation happens
generator = Generator()
generator.generate_ensemble(experiment.exp_path,
                            [high_res_knl, high_res_iv24])

# start the two high resolution models on the IV24 and KNL
# partitions.
experiment.start(ensembles=[high_res_knl, high_res_iv24])
experiment.poll(verbose=False)

# print out the statuses of the model we just ran
iv24_statuses = experiment.get_status(high_res_iv24)
print(f"Statuses of IV24 Models: {iv24_statuses}")

knl_statuses = experiment.get_status(high_res_knl)
print(f"Statuses of KNL Models: {knl_statuses}")

# Release the KNL partition because we dont need it anymore
experiment.release(alloc_id=knl_alloc)



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
    "ppn": 48,
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
generator.generate_ensemble(experiment.exp_path, low_res_iv24)


# start the low resolution simulation on the same
# allocation as the IV24 high resolution model
experiment.start(ensembles=low_res_iv24)
experiment.poll(verbose=False)

# print the statuses of the low resolution ensemble
# after it has completed.
iv24_low_res_statuses = experiment.get_status(low_res_iv24)
print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

# Release the iv24 partition
experiment.release(alloc_id=iv24_alloc)