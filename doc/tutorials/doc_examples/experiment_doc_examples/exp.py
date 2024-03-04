from smartsim import Experiment
from smartsim.log import get_logger

# Initialize an Experiment
exp = Experiment("example-experiment", launcher="auto")
# Initialize a SmartSim logger
smartsim_logger = get_logger("logger")

# Initialize an Orchestrator
standalone_database = exp.create_database(db_nodes=3, port=6379, interface="ib0")

# Initialize the Model RunSettings
settings = exp.create_run_settings("echo", exe_args="Hello World")
# Initialize the Model
model = exp.create_model("hello_world", settings)

# Generate the output directory
exp.generate(standalone_database, model, overwrite=True)

# Launch the Orchestrator then Model instance
exp.start(standalone_database, model)

# Clobber the Orchestrator
exp.stop(standalone_database)
# Log the summary of the Experiment
smartsim_logger.info(exp.summary())