import numpy as np
from smartredis import Client
from smartsim import Experiment
from smartsim.log import get_logger
import sys

exe_ex = sys.executable
logger = get_logger("Multidb Experiment Log")
# Initialize the Experiment
exp = Experiment("getting-started-multidb", launcher="auto")

# Initialize a single sharded database
single_shard_db = exp.create_database(port=6379, db_nodes=1, interface="ib0", db_identifier="single_shard_db_identifier")
exp.generate(single_shard_db, overwrite=True)

# Initialize a multi sharded database
multi_shard_db = exp.create_database(port=6380, db_nodes=3, interface="ib0", db_identifier="multi_shard_db_identifier")
exp.generate(multi_shard_db, overwrite=True)

# Launch the single and multi sharded database
exp.start(single_shard_db, multi_shard_db, summary=True)

# Initialize SmartRedis client for single sharded database
driver_client_single_shard = Client(cluster=False, address=single_shard_db.get_address()[0], logger_name="Single shard db logger")
# Initialize SmartRedis client for multi sharded database
driver_client_multi_shard = Client(cluster=True, address=multi_shard_db.get_address()[0], logger_name="Multi shard db logger")

# Create NumPy array
array_1 = np.array([1, 2, 3, 4])
# Use single shard db SmartRedis client to place tensor in single sharded db
driver_client_single_shard.put_tensor("tensor_1", array_1)

# Create NumPy array
array_2 = np.array([5, 6, 7, 8])
# Use single shard db SmartRedis client to place tensor in multi sharded db
driver_client_multi_shard.put_tensor("tensor_2", array_2)

# Check that tensors are in correct databases
check_single_shard_db_tensor_incorrect = driver_client_single_shard.key_exists("tensor_2")
check_multi_shard_db_tensor_incorrect = driver_client_multi_shard.key_exists("tensor_1")
logger.info(f"The multi shard array key exists in the incorrect database: {check_single_shard_db_tensor_incorrect}")
logger.info(f"The single shard array key exists in the incorrect database: {check_multi_shard_db_tensor_incorrect}")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe=exe_ex, exe_args="./path/to/application_script.py")
# Configure RunSettings object
model_settings.set_nodes(1)
model_settings.set_tasks_per_node(1)
# Initialize a SmartSim Model
model = exp.create_model("colo_model", model_settings)
# Colocate the Model
model.colocate_db_tcp(db_identifier="colo_db_identifier")
# Launch the colocated Model
exp.start(model, block=True, summary=True)

# Tear down the single and multi sharded databases
exp.stop(single_shard_db, multi_shard_db)
# Print the Experiment summary
logger.info(exp.summary())