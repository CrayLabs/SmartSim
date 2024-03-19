from smartredis import Client, LLInfo

# Initialize a Client
client = Client(cluster=False)

# Set the data source
client.set_data_source("producer_0")
# Check if the tensor exists
tensor_1 = client.poll_tensor("tensor", 100, 100)

# Set the data source
client.set_data_source("producer_1")
# Check if the tensor exists
tensor_2 = client.poll_tensor("tensor", 100, 100)

client.log_data(LLInfo, f"producer_0.tensor was found: {tensor_1}")
client.log_data(LLInfo, f"producer_1.tensor was found: {tensor_2}")