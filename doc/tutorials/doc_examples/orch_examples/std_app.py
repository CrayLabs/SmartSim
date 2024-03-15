from smartredis import Client, LLInfo
import numpy as np

# Initialize a SmartRedis Client
application_client = Client(cluster=True)

# Retrieve the driver script tensor from Orchestrator
driver_script_tensor = application_client.get_tensor("tensor_1")
# Log the tensor
application_client.log_data(LLInfo, f"The multi-sharded db tensor is: {driver_script_tensor}")

# Create a NumPy array
local_array = np.array([5, 6, 7, 8])
# Use SmartRedis client to place tensor in multi-sharded db
application_client.put_tensor("tensor_2", local_array)
