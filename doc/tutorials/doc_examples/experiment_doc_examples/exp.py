from smartsim import Experiment
from smartsim.log import get_logger

# Initialize an Experiment
exp = Experiment("example-experiment", launcher="auto")
# Initialize a SmartSim logger
smartsim_logger = get_logger("logger")

# Initialize an Feature Store
standalone_feature_store = exp.create_feature_store(fs_nodes=3, port=6379, interface="ib0")

# Initialize the Model RunSettings
settings = exp.create_run_settings("echo", exe_args="Hello World")
# Initialize the Model
model = exp.create_model("hello_world", settings)

# Generate the output directory
exp.generate(standalone_feature_store, model, overwrite=True)

# Launch the Feature Store then Model instance
exp.start(standalone_feature_store, model)

# Clobber the Feature Store
exp.stop(standalone_feature_store)
# Log the summary of the Experiment
smartsim_logger.info(exp.summary())