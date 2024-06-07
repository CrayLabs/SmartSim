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

torch1 = torch.zeros((3, 2, 5), dtype=torch.int8)
torch2 = torch.ones((1040, 1040, 3), dtype=torch.int64)
tflow1 = tf.zeros((3, 2, 5), dtype=tf.int8)
tflow2 = tf.ones((1040, 1040, 3), dtype=tf.int64)

tensor_1 = handler.build_tensor(torch1, "c", "int8", list(torch1.shape))
tensor_2 = handler.build_tensor(torch2, "c", "int64", list(torch2.shape))
tensor_3 = handler.build_tensor(tflow1, "c", "int8", list(tflow1.shape))
tensor_4 = handler.build_tensor(tflow2, "c", "int64", list(tflow2.shape))


result_key1 = handler.build_tensor_key("result_key1")
result_key2 = handler.build_tensor_key("result_key2")

torch_attributes = handler.build_torch_response_attributes()
tf_attributes = handler.build_tf_response_attributes()

indirect_response = handler.build_response(
    "complete",
    "Success!",
    [result_key1, result_key2],
    tf_attributes,
)
direct_response = handler.build_response(
    "complete",
    "Success again!",
    [tensor_1, tensor_2],
    torch_attributes,
)


@pytest.mark.parametrize(
    "status, status_message, result, custom_attribute",
    [
        pytest.param(
            200,
            "Yay, it worked!",
            [tensor_3, tensor_4],
            None,
            id="tensor list",
        ),
        pytest.param(
            200,
            "Yay, it worked!",
            [tensor_1],
            torch_attributes,
            id="small tensor",
        ),
        pytest.param(
            200,
            "Yay, it worked!",
            [result_key1, result_key2],
            tf_attributes,
            id="tensor key list",
        ),
    ],
)
def test_build_response_successful(status, status_message, result, custom_attribute):
    response = handler.build_response(
        status=status,
        message=status_message,
        result=result,
        custom_attributes=custom_attribute,
    )
    assert response is not None
    assert response.status == status
    assert response.message == status_message
    if response.result.which() == "keys":
        assert response.result.keys[0].to_dict() == result[0].to_dict()
    else:
        assert response.result.data[0].to_dict() == result[0].to_dict()


@pytest.mark.parametrize(
    "status, status_message, result, custom_attribute",
    [
        pytest.param(
            "bad status",
            "Yay, it worked!",
            [tensor_3, tensor_4],
            None,
            id="bad status",
        ),
        pytest.param(
            "complete",
            200,
            [tensor_1],
            torch_attributes,
            id="bad status message",
        ),
        pytest.param(
            "complete",
            "Yay, it worked!",
            ["result_key1", "result_key2"],
            tf_attributes,
            id="bad result",
        ),
        pytest.param(
            "complete",
            "Yay, it worked!",
            [tf_attributes],
            tf_attributes,
            id="bad result type",
        ),
        pytest.param(
            "complete",
            "Yay, it worked!",
            [tensor_3, tensor_4],
            "custom attributes",
            id="bad custom attributes",
        ),
        pytest.param(
            "complete",
            "Yay, it worked!",
            [tensor_3, tensor_4],
            tensor_1,
            id="bad custom attributes type",
        ),
    ],
)
def test_build_response_unsuccessful(status, status_message, result, custom_attribute):
    with pytest.raises(ValueError):
        response = handler.build_response(
            status, status_message, result, custom_attribute
        )


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(indirect_response, id="indirect"),
        pytest.param(direct_response, id="direct"),
    ],
)
def test_serialize_response(response):
    serialized = handler.serialize_response(response)
    assert type(serialized) == bytes

    deserialized = handler.deserialize_response(serialized)
    assert deserialized.to_dict() == response.to_dict()
