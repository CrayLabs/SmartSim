# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
