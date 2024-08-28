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

from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

handler = MessageHandler()

fsd = "mock-feature-store-descriptor"
tensor_key = handler.build_feature_store_key("key", fsd)


@pytest.mark.parametrize(
    "order, keys, dtype, dimension",
    [
        pytest.param("c", [tensor_key], "int8", [1, 2, 3, 4], id="all specified"),
        pytest.param(
            "c", [tensor_key, tensor_key], "none", [1, 2, 3, 4], id="none dtype"
        ),
        pytest.param("c", [tensor_key], "int8", [], id="empty dimensions"),
        pytest.param("c", [], "int8", [1, 2, 3, 4], id="empty keys"),
    ],
)
def test_build_output_tensor_descriptor_successful(dtype, keys, order, dimension):
    built_descriptor = handler.build_output_tensor_descriptor(
        order, keys, dtype, dimension
    )
    assert built_descriptor is not None
    assert built_descriptor.order == order
    assert len(built_descriptor.optionalKeys) == len(keys)
    assert built_descriptor.optionalDatatype == dtype
    for i, j in zip(built_descriptor.optionalDimension, dimension):
        assert i == j


@pytest.mark.parametrize(
    "order, keys, dtype, dimension",
    [
        pytest.param("bad_order", [], "int8", [3, 2, 5], id="bad order type"),
        pytest.param(
            "f", [tensor_key], "bad_num_type", [3, 2, 5], id="bad numerical type"
        ),
        pytest.param("f", [tensor_key], "int8", "bad shape type", id="bad shape type"),
        pytest.param("f", ["tensor_key"], "int8", [3, 2, 5], id="bad key type"),
    ],
)
def test_build_output_tensor_descriptor_unsuccessful(order, keys, dtype, dimension):
    with pytest.raises(ValueError):
        built_tensor = handler.build_output_tensor_descriptor(
            order, keys, dtype, dimension
        )
