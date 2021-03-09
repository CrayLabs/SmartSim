import os
import io

import numpy as np
import torch
import torch.nn as nn

import argparse

from silc import Client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SILC ensemble producer process.')
    parser.add_argument('--exchange', action='store_true')
    args = parser.parse_args()

    # get model and set into database
    c = Client(False)

    keyin = os.getenv("SSKEYIN")
    print(keyin)
    data_sources = keyin.split(',')
    data_sources.sort()

    assert data_sources == ['producer_0', 'producer_1']

    input_sum = np.zeros((1,1,3,3))

    for key in data_sources:
        c.set_data_source(key)

        input_exists = c.poll_tensor("torch_cnn_input", 100, 100)
        assert input_exists
        input_sum += c.get_tensor("torch_cnn_input")

    # One process will send 1s, the other -10s. Sum is -9s.
    assert np.sum(input_sum)==-9.0*input_sum.size

    for key in data_sources:
        c.set_data_source(key)
        output_exists = c.poll_tensor("torch_cnn_output", 100, 100)
        assert output_exists

        out_data = c.get_tensor("torch_cnn_output")
        assert out_data.shape == (1, 1, 1, 1)
