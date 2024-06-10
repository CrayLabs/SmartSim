# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import pytest

try:
    import tensorflow as tf
except ImportError:
    should_run_tf = False
else:
    should_run_tf = True


try:
    import torch
except ImportError:
    should_run_torch = False
else:
    should_run_torch = True

from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

handler = MessageHandler()


@pytest.mark.skipif(not should_run_torch, reason="Test needs Torch to run")
@pytest.mark.parametrize(
    "tensor, dtype, order, dimension",
    [
        pytest.param(
            torch.zeros((3, 2, 5), dtype=torch.int8),
            "int8",
            "c",
            [3, 2, 5],
            id="small torch tensor",
        ),
        pytest.param(
            torch.ones((1040, 1040, 3), dtype=torch.int64),
            "int64",
            "c",
            [1040, 1040, 3],
            id="medium torch tensor",
        ),
    ],
)
def test_build_torch_tensor_successful(tensor, dtype, order, dimension):
    built_tensor = handler.build_tensor(tensor, order, dtype, dimension)
    assert built_tensor is not None
    assert type(built_tensor.blob) == bytes
    assert built_tensor.tensorDescriptor.order == order
    assert built_tensor.tensorDescriptor.dataType == dtype
    for i, j in zip(built_tensor.tensorDescriptor.dimensions, dimension):
        assert i == j


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
@pytest.mark.parametrize(
    "tensor, dtype, order, dimension",
    [
        pytest.param(
            tf.zeros((3, 2, 5), dtype=tf.int8),
            "int8",
            "c",
            [3, 2, 5],
            id="small tf tensor",
        ),
        pytest.param(
            tf.ones((1040, 1040, 3), dtype=tf.int64),
            "int64",
            "c",
            [1040, 1040, 3],
            id="medium tf tensor",
        ),
    ],
)
def test_build_tf_tensor_successful(tensor, dtype, order, dimension):
    built_tensor = handler.build_tensor(tensor, order, dtype, dimension)
    assert built_tensor is not None
    assert type(built_tensor.blob) == bytes
    assert built_tensor.tensorDescriptor.order == order
    assert built_tensor.tensorDescriptor.dataType == dtype
    for i, j in zip(built_tensor.tensorDescriptor.dimensions, dimension):
        assert i == j


@pytest.mark.skipif(not should_run_torch, reason="Test needs Torch to run")
@pytest.mark.parametrize(
    "tensor, dtype, order, dimension",
    [
        pytest.param([1, 2, 4], "c", "int8", [1, 2, 3], id="bad tensor type"),
        pytest.param(
            torch.zeros((3, 2, 5), dtype=torch.int8),
            "bad_order",
            "int8",
            [3, 2, 5],
            id="bad order type",
        ),
        pytest.param(
            torch.zeros((3, 2, 5), dtype=torch.int8),
            "f",
            "bad_num_type",
            [3, 2, 5],
            id="bad numerical type",
        ),
        pytest.param(
            torch.zeros((3, 2, 5), dtype=torch.int8),
            "f",
            "int8",
            "bad shape type",
            id="bad shape type",
        ),
    ],
)
def test_build_torch_tensor_bad_input(tensor, dtype, order, dimension):
    with pytest.raises(ValueError):
        built_tensor = handler.build_tensor(tensor, order, dtype, dimension)


@pytest.mark.skipif(not should_run_tf, reason="Test needs TF to run")
@pytest.mark.parametrize(
    "tensor, dtype, order, dimension",
    [
        pytest.param([1, 2, 4], "c", "int8", [1, 2, 3], id="bad tensor type"),
        pytest.param(
            tf.zeros((3, 2, 5), dtype=tf.int8),
            "bad_order",
            "int8",
            [3, 2, 5],
            id="bad order type",
        ),
        pytest.param(
            tf.zeros((3, 2, 5), dtype=tf.int8),
            "f",
            "bad_num_type",
            [3, 2, 5],
            id="bad numerical type",
        ),
        pytest.param(
            tf.zeros((3, 2, 5), dtype=tf.int8),
            "f",
            "int8",
            "bad shape type",
            id="bad shape type",
        ),
    ],
)
def test_build_tf_tensor_bad_input(tensor, dtype, order, dimension):
    with pytest.raises(ValueError):
        built_tensor = handler.build_tensor(tensor, order, dtype, dimension)
