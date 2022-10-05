import numpy as np
from smartredis import Client

def main():
    # Address should be set as we are launching through
    # SmartSim.
    client = Client(cluster=False)

    array = np.ones((1, 1, 28, 28)).astype(np.single)
    client.put_tensor("test_array", array)
    assert client.poll_model("cnn", 500, 30)
    client.run_model("cnn", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")

    assert returned.shape == (1, 10)

    print(f"Test worked!")

if __name__ == "__main__":
    main()
