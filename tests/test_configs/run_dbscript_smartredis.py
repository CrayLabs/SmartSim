import numpy as np
from smartredis import Client
from pytest import approx

def main():
    # address should be set as we are launching through
    # SmartSim.
    client = Client(cluster=False)

    array = np.ones((1, 3, 3, 1)).astype(np.single)
    client.put_tensor("test_array", array)
    assert client.poll_model("test_script1", 500, 30)
    client.run_script("test_script1", "average", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")
    assert returned == approx(np.mean(array))

    assert client.poll_model("test_script2", 500, 30)
    client.run_script("test_script2", "negate", ["test_array"], ["test_output"])
    returned = client.get_tensor("test_output")

    assert returned == approx(-array)

    if client.model_exists("test_func"):
        client.run_script("test_func", "timestwo", ["test_array"], ["test_output"])
        returned = client.get_tensor("test_output")
        assert returned == approx(2*array)

    print(f"Test worked!")

if __name__ == "__main__":
    main()