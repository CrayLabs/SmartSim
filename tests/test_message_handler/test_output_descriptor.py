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
import tensorflow as tf
import torch

from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

handler = MessageHandler()


@pytest.mark.parametrize(
    "order, dtype, dimension",
    [
        pytest.param("c", "int8", [1, 2, 3, 4], id="specified dtype and dimension"),
        pytest.param("c", None, [1, 2, 3, 4], id="specified dimension"),
        pytest.param("c", "int8", None, id="specified dtype"),
    ],
)
def test_build_output_tensor_descriptor_successful(dtype, order, dimension):
    built_descriptor = handler.build_output_tensor_descriptor(order, dtype, dimension)
    assert built_descriptor is not None
    assert built_descriptor.order == order
    if built_descriptor.optionalDatatype.which() == "dataType":
        assert built_descriptor.optionalDatatype.dataType == dtype
    else:
        assert built_descriptor.optionalDatatype.none == dtype
    if built_descriptor.optionalDimension.which() == "dimensions":
        for i, j in zip(built_descriptor.optionalDimension.dimensions, dimension):
            assert i == j
    else:
        assert built_descriptor.optionalDimension.none == dimension
