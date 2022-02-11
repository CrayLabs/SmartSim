import numpy as np
from smartredis import Client

def main():
    # address should be set as we are launching through
    # SmartSim.

    client = Client(cluster=False)

    array = np.array([1,2,3,4])
    client.put_tensor("test_array", array)
    returned = client.get_tensor("test_array")

    np.testing.assert_array_equal(array, returned)
    print(f"Test worked! Sent and received array: {str(array)}")

if __name__ == "__main__":
    main()