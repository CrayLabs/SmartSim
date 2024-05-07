from smartredis import ConfigOptions, Client
from smartredis import *
from smartredis.error import *

# Initialize a ConfigOptions object
single_shard_config = ConfigOptions.create_from_environment("single_shard_fs_identifier")
# Initialize a SmartRedis client for the single sharded feature store
app_single_shard_client = Client(single_shard_config, logger_name="Model: single shard logger")

# Initialize a ConfigOptions object
multi_shard_config = ConfigOptions.create_from_environment("multi_shard_fs_identifier")
# Initialize a SmartRedis client for the multi sharded feature store
app_multi_shard_client = Client(multi_shard_config, logger_name="Model: multi shard logger")

# Initialize a ConfigOptions object
colo_config = ConfigOptions.create_from_environment("colo_fs_identifier")
# Initialize a SmartRedis client for the colocated feature store
colo_client = Client(colo_config, logger_name="Model: colo logger")

# Retrieve the tensor placed in driver script using the associated client
val1 = app_single_shard_client.get_tensor("tensor_1")
val2 = app_multi_shard_client.get_tensor("tensor_2")

# Print message to stdout using SmartRedis Client logger
app_single_shard_client.log_data(LLInfo, f"The single sharded fs tensor is: {val1}")
app_multi_shard_client.log_data(LLInfo, f"The multi sharded fs tensor is: {val2}")

# Place retrieved tensors in colocated feature store
colo_client.put_tensor("tensor_1", val1)
colo_client.put_tensor("tensor_2", val2)

# Check that tensors are in colocated feature store
colo_val1 = colo_client.poll_tensor("tensor_1", 10, 10)
colo_val2 = colo_client.poll_tensor("tensor_2", 10, 10)
# Print message to stdout using SmartRedis Client logger
colo_client.log_data(LLInfo, f"The colocated fs has tensor_1: {colo_val1}")
colo_client.log_data(LLInfo, f"The colocated fs has tensor_2: {colo_val2}")