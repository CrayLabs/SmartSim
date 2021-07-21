
import sys
import io
import pytest
import numpy as np
from smartsim import Experiment
from smartsim.database import Orchestrator


try:
    import torch
    import torch.nn as nn
    from smartredis import Client
    from smartredis.error import RedisReplyError
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    ("torch" not in sys.modules),
    reason="requires PyTorch",
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

def create_torch_model():
    n = Net()
    example_forward_input = torch.rand(1, 1, 3, 3)
    module = torch.jit.trace(n, example_forward_input)
    model_buffer = io.BytesIO()
    torch.jit.save(module, model_buffer)
    return model_buffer.getvalue()

# random function from TorchScript API
def calc_svd(input_tensor):
    return input_tensor.svd()

def test_torch_model_and_script(fileutils):

    exp_name = 'test_torch_script_svd'
    exp = Experiment("exp_name", launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(port=6780)
    db.set_path(test_dir)
    exp.start(db)

    test_status = True
    try:
        # connect a client to the database
        client = Client(address="127.0.0.1:6780",
                        cluster=False)


        # test the SVD function
        tensor = np.random.randint(0, 100,
                                size=(5, 3, 2)).astype(np.float32)
        client.put_tensor("input", tensor)
        client.set_function("svd", calc_svd)
        client.run_script("svd", "calc_svd",
                        "input", ["U", "S", "V"])
        U = client.get_tensor("U")
        S = client.get_tensor("S")
        V = client.get_tensor("V")
        print(f"U: {U}, S: {S}, V: {V}")


        # test simple convNet
        net = create_torch_model()
        example_forward_input = torch.rand(1, 1, 3, 3)
        client.set_model("cnn", net, "TORCH", device="CPU")
        client.put_tensor("input", example_forward_input.numpy())
        client.run_model("cnn", inputs=["input"], outputs=["output"])

        output = client.get_tensor("output")
        print(f"Prediction: {output}")

    except RedisReplyError as e:
        print("Caught Database error")
        print(e)
        test_status = False

    finally:
        exp.stop(db)
        assert(test_status)
