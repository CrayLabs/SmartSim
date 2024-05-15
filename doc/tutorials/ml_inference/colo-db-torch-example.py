import numpy as np
from smartredis import Client

def calc_svd(input_tensor):
    # svd function from TorchScript API
    # torch isn't imported since we don't need that dependency
    # in the client code to call this function in the database.
    return input_tensor.svd()


# connect a client to the database
# no address required since this `Model` was launched through SmartSim
# Cluster=False since colocated databases are never clustered.
client = Client(cluster=False)

tensor = np.random.randint(0, 100, size=(5, 3, 2)).astype(np.float32)
client.put_tensor("input", tensor)
client.set_function("svd", calc_svd)
client.run_script("svd", "calc_svd", ["input"], ["U", "S", "V"])

U = client.get_tensor("U")
S = client.get_tensor("S")
V = client.get_tensor("V")

print(f"U: {U}\n\n, S: {S}\n\n, V: {V}\n")