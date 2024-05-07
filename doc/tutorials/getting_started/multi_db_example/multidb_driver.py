import numpy as np
from smartredis import Client
from smartsim import Experiment
from smartsim.log import get_logger
import sys

exe_ex = sys.executable
logger = get_logger("MultiFS Experiment Log")
# Initialize the Experiment
exp = Experiment("getting-started-multifs", launcher="auto")

# Initialize a single sharded feature store
single_shard_fs = exp.create_feature_store(port=6379, fs_nodes=1, interface="ib0", fs_identifier="single_shard_fs_identifier")
exp.generate(single_shard_fs, overwrite=True)

# Initialize a multi sharded feature store
multi_shard_fs = exp.create_feature_store(port=6380, fs_nodes=3, interface="ib0", fs_identifier="multi_shard_fs_identifier")
exp.generate(multi_shard_fs, overwrite=True)

# Launch the single and multi sharded feature store
exp.start(single_shard_fs, multi_shard_fs, summary=True)

# Initialize SmartRedis client for single sharded feature store
driver_client_single_shard = Client(cluster=False, address=single_shard_fs.get_address()[0], logger_name="Single shard fs logger")
# Initialize SmartRedis client for multi sharded feature store
driver_client_multi_shard = Client(cluster=True, address=multi_shard_fs.get_address()[0], logger_name="Multi shard fs logger")

# Create NumPy array
array_1 = np.array([1, 2, 3, 4])
# Use single shard fs SmartRedis client to place tensor in single sharded fs
driver_client_single_shard.put_tensor("tensor_1", array_1)

# Create NumPy array
array_2 = np.array([5, 6, 7, 8])
# Use single shard fs SmartRedis client to place tensor in multi sharded fs
driver_client_multi_shard.put_tensor("tensor_2", array_2)

# Check that tensors are in correct feature stores
check_single_shard_fs_tensor_incorrect = driver_client_single_shard.key_exists("tensor_2")
check_multi_shard_fs_tensor_incorrect = driver_client_multi_shard.key_exists("tensor_1")
logger.info(f"The multi shard array key exists in the incorrect feature store: {check_single_shard_fs_tensor_incorrect}")
logger.info(f"The single shard array key exists in the incorrect feature store: {check_multi_shard_fs_tensor_incorrect}")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe=exe_ex, exe_args="./path/to/application_script.py")
# Configure RunSettings object
model_settings.set_nodes(1)
model_settings.set_tasks_per_node(1)
# Initialize a SmartSim Model
model = exp.create_model("colo_model", model_settings)
# Colocate the Model
model.colocate_fs_tcp(fs_identifier="colo_fs_identifier")
# Launch the colocated Model
exp.start(model, block=True, summary=True)

# Tear down the single and multi sharded feature stores
exp.stop(single_shard_fs, multi_shard_fs)
# Print the Experiment summary
logger.info(exp.summary())