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

model_key = handler.build_model_key("model_key")

input_key1 = handler.build_tensor_key("input_key1")
input_key2 = handler.build_tensor_key("input_key2")

output_key1 = handler.build_tensor_key("output_key1")
output_key2 = handler.build_tensor_key("output_key2")

torch_attributes = handler.build_torch_request_attributes("sparse")
tf_attributes = handler.build_tf_request_attributes(name="tf", tensor_type="sparse")

indirect_request = handler.build_request(
    b"reply",
    b"model",
    "cpu",
    [input_key1, input_key2],
    [output_key1, output_key2],
    tf_attributes,
)
direct_request = handler.build_request(
    b"reply",
    b"model",
    "cpu",
    [tensor_1, tensor_2],
    [tensor_3, tensor_4],
    torch_attributes,
)


@pytest.mark.parametrize(
    "reply_channel, model, device, input, output, custom_attributes",
    [
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1, input_key2],
            [output_key1, output_key2],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            b"model data",
            "gpu",
            [input_key1],
            [output_key2],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            b"model data",
            None,
            [input_key1],
            [output_key2],
            tf_attributes,
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1],
            [output_key2],
            None,
        ),
    ],
)
def test_build_request_indirect_successful(
    reply_channel, model, device, input, output, custom_attributes
):
    built_request = handler.build_request(
        reply_channel, model, device, input, output, custom_attributes
    )
    assert built_request is not None
    assert built_request.replyChannel.reply == reply_channel
    if built_request.model.which() == "modelKey":
        assert built_request.model.modelKey.key == model.key
    else:
        assert built_request.model.modelData == model
    if built_request.device.which() == "deviceType":
        assert built_request.device.deviceType == device
    else:
        assert built_request.device.noDevice == device
    assert built_request.input.which() == "inputKeys"
    assert built_request.input.inputKeys[0].key == input[0].key
    assert len(built_request.input.inputKeys) == len(input)
    assert built_request.output.which() == "outputKeys"
    assert len(built_request.output.outputKeys) == len(input)
    print(built_request.customAttributes.which())
    if built_request.customAttributes.which() == "tf":
        assert (
            built_request.customAttributes.tf.tensorType == custom_attributes.tensorType
        )
    elif built_request.customAttributes.which() == "torch":
        assert (
            built_request.customAttributes.torch.tensorType
            == custom_attributes.tensorType
        )
    else:
        assert built_request.customAttributes.none == custom_attributes


@pytest.mark.parametrize(
    "reply_channel, model, device, input, output, custom_attributes",
    [
        pytest.param(
            [],
            model_key,
            "cpu",
            [input_key1, input_key2],
            [output_key1, output_key2],
            torch_attributes,
            id="bad channel",
        ),
        pytest.param(
            b"reply channel",
            "bad model",
            "gpu",
            [input_key1],
            [output_key2],
            torch_attributes,
            id="bad model",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "bad device",
            [input_key1],
            [output_key2],
            torch_attributes,
            id="bad device",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            ["input_key1", "input_key2"],
            [output_key1, output_key2],
            torch_attributes,
            id="bad inputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [model_key],
            [output_key1, output_key2],
            torch_attributes,
            id="bad input schema type",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1],
            ["output_key1", "output_key2"],
            torch_attributes,
            id="bad outputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1],
            [model_key],
            torch_attributes,
            id="bad output schema type",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1],
            [output_key1, output_key2],
            "bad attributes",
            id="bad custom attributes",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [input_key1],
            [output_key1, output_key2],
            model_key,
            id="bad custom attributes schema type",
        ),
    ],
)
def test_build_request_indirect_unsuccessful(
    reply_channel, model, device, input, output, custom_attributes
):
    with pytest.raises(ValueError):
        built_request = handler.build_request(
            reply_channel, model, device, input, output, custom_attributes
        )


@pytest.mark.parametrize(
    "reply_channel, model, device, input, output, custom_attributes",
    [
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [tensor_1, tensor_2],
            [tensor_3, tensor_4],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            b"model data",
            "gpu",
            [tensor_1],
            [tensor_2],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            b"model data",
            None,
            [tensor_3],
            [tensor_4],
            tf_attributes,
        ),
        pytest.param(
            b"another reply channel",
            b"model data",
            None,
            [tensor_3],
            [tensor_4],
            None,
        ),
    ],
)
def test_build_request_direct_successful(
    reply_channel, model, device, input, output, custom_attributes
):
    built_request = handler.build_request(
        reply_channel, model, device, input, output, custom_attributes
    )
    assert built_request is not None
    assert built_request.replyChannel.reply == reply_channel
    if built_request.model.which() == "modelKey":
        assert built_request.model.modelKey.key == model.key
    else:
        assert built_request.model.modelData == model
    if built_request.device.which() == "deviceType":
        assert built_request.device.deviceType == device
    else:
        assert built_request.device.noDevice == device
    assert built_request.input.which() == "inputData"
    assert built_request.input.inputData[0].blob == input[0].blob
    assert len(built_request.input.inputData) == len(input)
    assert built_request.output.which() == "outputData"
    assert len(built_request.output.outputData) == len(input)
    if built_request.customAttributes.which() == "tf":
        assert (
            built_request.customAttributes.tf.tensorType == custom_attributes.tensorType
        )
    elif built_request.customAttributes.which() == "torch":
        assert (
            built_request.customAttributes.torch.tensorType
            == custom_attributes.tensorType
        )
    else:
        assert built_request.customAttributes.none == custom_attributes


@pytest.mark.parametrize(
    "reply_channel, model, device, input, output, custom_attributes",
    [
        pytest.param(
            [],
            model_key,
            "cpu",
            [tensor_1, tensor_2],
            [tensor_3, tensor_4],
            torch_attributes,
            id="bad channel",
        ),
        pytest.param(
            b"reply channel",
            "bad model",
            "gpu",
            [tensor_1],
            [tensor_3],
            torch_attributes,
            id="bad model",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "bad device",
            [tensor_2],
            [tensor_1],
            torch_attributes,
            id="bad device",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            ["input_key1", "input_key2"],
            [tensor_1, tensor_2],
            torch_attributes,
            id="bad inputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [tensor_4],
            ["output_key1", "output_key2"],
            torch_attributes,
            id="bad outputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            "cpu",
            [tensor_1],
            [tensor_2, tensor_3],
            "bad attributes",
            id="bad custom attributes",
        ),
    ],
)
def test_build_request_direct_unsuccessful(
    reply_channel, model, device, input, output, custom_attributes
):
    with pytest.raises(ValueError):
        built_request = handler.build_request(
            reply_channel, model, device, input, output, custom_attributes
        )


@pytest.mark.parametrize(
    "req",
    [
        pytest.param(indirect_request, id="indirect"),
        pytest.param(direct_request, id="direct"),
    ],
)
def test_serialize_request_successful(req):
    serialized = handler.serialize_request(req)
    assert type(serialized) == bytes

    deserialized = handler.deserialize_request(serialized)
    assert deserialized.to_dict() == req.to_dict()
