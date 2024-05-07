import numpy as np
from smartredis import Client
from smartsim import Experiment
from smartsim.log import get_logger

# Initialize the logger
logger = get_logger("Example Experiment Log")
# Initialize the Experiment
exp = Experiment("getting-started", launcher="auto")

# Initialize a multi-sharded feature store
standalone_feature_store = exp.create_feature_store(fs_nodes=3)

# Initialize a SmartRedis client for multi-sharded feature store
driver_client = Client(cluster=True, address=standalone_feature_store.get_address()[0])

# Create NumPy array
local_array = np.array([1, 2, 3, 4])
# Use the SmartRedis client to place tensor in the standalone feature store
driver_client.put_tensor("tensor_1", local_array)

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="/path/to/executable_simulation")
model_settings.set_nodes(1)

# Initialize the Model
model = exp.create_model("model", model_settings)

# Create the output directory
exp.generate(standalone_feature_store, model)

# Launch the multi-sharded feature store
exp.start(standalone_feature_store)

# Launch the Model
exp.start(model, block=True, summary=True)

# Poll the tensors placed by the Model
app_tensor = driver_client.poll_key("tensor_2", 100, 10)
# Validate that the tensor exists
logger.info(f"The tensor exists: {app_tensor}")

# Cleanup the feature store
exp.stop(standalone_feature_store)
# Print the Experiment summary
logger.info(exp.summary())