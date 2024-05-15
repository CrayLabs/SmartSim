import numpy as np
from smartredis import Client
from smartsim import Experiment
from smartsim.log import get_logger

# Initialize a logger object
logger = get_logger("Example Experiment Log")
# Initialize the Experiment
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="path/to/executable_simulation")
# Configure RunSettings object
model_settings.set_nodes(1)

# Initialize a SmartSim Model
model = exp.create_model("colo_model", model_settings)

# Colocate the Model
model.colocate_db_uds()

# Generate output files
exp.generate(model)

# Launch the colocated Model
exp.start(model, block=True, summary=True)

# Log the Experiment summary
logger.info(exp.summary())