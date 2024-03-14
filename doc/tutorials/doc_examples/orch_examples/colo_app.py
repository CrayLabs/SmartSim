from smartredis import Client, LLInfo
import numpy as np

# Initialize a Client
colo_client = Client(cluster=False)

# Create NumPy array
local_array = np.array([1, 2, 3, 4])
# Store the NumPy tensor
colo_client.put_tensor("tensor_1", local_array)

# Retrieve tensor from driver script
local_tensor = colo_client.get_tensor("tensor_1")
# Log tensor
colo_client.log_data(LLInfo, f"The colocated db tensor is: {local_tensor}")