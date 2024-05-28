import numpy as np
from smartredis import Client

# Initialize a Client
client = Client(cluster=False)

# Create NumPy array
array = np.array([1, 2, 3, 4])
# Use SmartRedis Client to place tensor in standalone Orchestrator
client.put_tensor("tensor", array)