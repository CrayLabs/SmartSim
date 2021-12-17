import argparse
import io
import os

import numpy as np
import torch
import torch.nn as nn

from smartredis import Client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartRedis ensemble consumer process."
    )
    parser.add_argument("--exchange", action="store_true")
    args = parser.parse_args()

    # get model and set into database
    c = Client(False)

    keyin = os.getenv("SSKEYIN")
    data_sources = keyin.split(",")
    data_sources.sort()

    assert data_sources == ["producer_0", "producer_1"]

    inputs = {
        data_sources[0]: np.ones((1, 1, 3, 3)),
        data_sources[1]: -np.ones((1, 1, 3, 3)),
    }

    for key in data_sources:
        c.set_data_source(key)

        input_exists = c.poll_tensor("torch_cnn_input", 100, 100)
        assert input_exists
        db_tensor = c.get_tensor("torch_cnn_input")
        assert np.all(db_tensor == inputs[key])

    for key in data_sources:
        c.set_data_source(key)
        output_exists = c.poll_tensor("torch_cnn_output", 100, 100)
        assert output_exists

        out_data = c.get_tensor("torch_cnn_output")
        assert out_data.shape == (1, 1, 1, 1)
