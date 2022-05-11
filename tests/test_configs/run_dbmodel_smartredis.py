import numpy as np
from smartredis import Client

def main():
    # address should be set as we are launching through
    # SmartSim.
    client = Client(cluster=False)

    array = np.ones((1, 3, 3, 1)).astype(np.single)
    client.put_tensor("test_array", array)
    assert client.poll_model("cnn", 500, 30)
    client.run_model("cnn", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")

    assert returned.shape == (1, 1, 1, 1)

    array = np.ones((1, 3, 3, 1)).astype(np.single)
    assert client.poll_model("cnn2", 500, 30)
    client.run_model("cnn2", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")

    assert returned.shape == (1, 1, 1, 1)
    print(f"Test worked!")

if __name__ == "__main__":
    main()