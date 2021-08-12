import numpy as np
from smartredis import Client


# send some arrays to the database cluster
# the following functions are largely the same across all the
# client languages: C++, C, Fortran, Python

# since we are launching through SmartSim, db address
# will be found automagically.
client = Client(address=None, cluster=True)

# put into database
test_array = np.array([1,2,3,4])
print(f"Array put in database: {test_array}")
client.put_tensor("test", test_array)

# get from database
returned_array = client.get_tensor("test")
print(f"Array retrieved from database: {returned_array}")
