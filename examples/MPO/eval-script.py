import xarray as xr
from smartsim import MPO

# initialize the fields needed by MPO in each
# evaluation run.
tunable_params = {"KH": 1000,
                  "KHTH": 1000}
model_params =  {"months": 12,
                 "x_resolution": 40,
                 "y_resolution": 40}
run_settings = {"nodes":1,
                "ppn": 48,
                "executable":"MOM6",
                "partition": "iv24"}

# intialize the MPO instance and name the data directory "MOM6-mpo"
mpo = MPO(tunable_params, data_dir="MOM6-mpo")

# initialize the model we want to evaluate.
# configure and copy needed model files into the
# directory where the evaluation model will be run.
model = mpo.init_model(run_settings, model_params=model_params)
model.attach_generator_files(
    to_copy=["../MOM6/MOM6_base_config/"],
    to_configure=["../MOM6/MOM6_base_config/input.nml",
                  "../MOM6/MOM6_base_config/MOM_input"])

# Start the underlying experiment that
# contains the generated and configured model
# we are optimizing.
mpo.run()

# get data produced by the simulation
data_path = mpo.get_model_file("ocean_mean_month.nc")
grid_path = mpo.get_model_file("ocean_geometry.nc")

# perform evaluation to calculate figure of merit
data = xr.open_dataset(data_path, decode_times=False)
grid = xr.open_dataset(grid_path,
                        decode_times=False).rename({'lonh' : 'xh',
                                                    'lath' : 'yh'})

# calculate MSE of jet penetration between the
# evaluated model and high resolution data which
# we will use at the figure of merit
num = (data.KE.sum("zl")*grid.geolon*grid.Ah).sum(("xh","yh"))
denom = (data.KE.sum("zl")*grid.Ah).sum(("xh","yh"))
jp = (num/denom).mean("time").values
fom = (jp - 17)**2 # 17 is a rough guess; squaring the error

# print figure of merit for CrayAI optimizer
print("FoM:", fom)

