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

fsd = "mock-feature-store-descriptor"

model_key = MessageHandler.build_model_key("model_key", fsd)
model = MessageHandler.build_model(b"model data", "model_name", "v0.0.1")

input_key1 = MessageHandler.build_tensor_key("input_key1", fsd)
input_key2 = MessageHandler.build_tensor_key("input_key2", fsd)

output_key1 = MessageHandler.build_tensor_key("output_key1", fsd)
output_key2 = MessageHandler.build_tensor_key("output_key2", fsd)

output_descriptor1 = MessageHandler.build_output_tensor_descriptor(
    "c", [output_key1, output_key2], "int64", []
)
output_descriptor2 = MessageHandler.build_output_tensor_descriptor("f", [], "auto", [])
output_descriptor3 = MessageHandler.build_output_tensor_descriptor(
    "c", [output_key1], "none", [1, 2, 3]
)
torch_attributes = MessageHandler.build_torch_request_attributes("sparse")
tf_attributes = MessageHandler.build_tf_request_attributes(
    name="tf", tensor_type="sparse"
)

tensor_1 = MessageHandler.build_tensor_descriptor("c", "int8", [1])
tensor_2 = MessageHandler.build_tensor_descriptor("c", "int64", [3, 2])
tensor_3 = MessageHandler.build_tensor_descriptor("f", "int8", [1])
tensor_4 = MessageHandler.build_tensor_descriptor("f", "int64", [3, 2])


tf_indirect_request = MessageHandler.build_request(
    b"reply",
    model,
    [input_key1, input_key2],
    [output_key1, output_key2],
    [output_descriptor1, output_descriptor2, output_descriptor3],
    tf_attributes,
)

tf_direct_request = MessageHandler.build_request(
    b"reply",
    model,
    [tensor_3, tensor_4],
    [],
    [output_descriptor1, output_descriptor2],
    tf_attributes,
)

torch_indirect_request = MessageHandler.build_request(
    b"reply",
    model,
    [input_key1, input_key2],
    [output_key1, output_key2],
    [output_descriptor1, output_descriptor2, output_descriptor3],
    torch_attributes,
)

torch_direct_request = MessageHandler.build_request(
    b"reply",
    model,
    [tensor_1, tensor_2],
    [],
    [output_descriptor1, output_descriptor2],
    torch_attributes,
)


@pytest.mark.parametrize(
    "reply_channel, model, input, output, output_descriptors, custom_attributes",
    [
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1, input_key2],
            [output_key1, output_key2],
            [output_descriptor1],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            model,
            [input_key1],
            [output_key2],
            [output_descriptor1],
            tf_attributes,
        ),
        pytest.param(
            b"another reply channel",
            model,
            [input_key1],
            [output_key2],
            [output_descriptor1],
            torch_attributes,
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            [output_key1],
            [output_descriptor1],
            None,
        ),
    ],
)
def test_build_request_indirect_successful(
    reply_channel, model, input, output, output_descriptors, custom_attributes
):
    built_request = MessageHandler.build_request(
        reply_channel,
        model,
        input,
        output,
        output_descriptors,
        custom_attributes,
    )
    assert built_request is not None
    assert built_request.replyChannel.descriptor == reply_channel
    if built_request.model.which() == "key":
        assert built_request.model.key.key == model.key
    else:
        assert built_request.model.data.data == model.data
        assert built_request.model.data.name == model.name
        assert built_request.model.data.version == model.version
    assert built_request.input.which() == "keys"
    assert built_request.input.keys[0].key == input[0].key
    assert len(built_request.input.keys) == len(input)
    assert len(built_request.output) == len(output)
    for i, j in zip(built_request.outputDescriptors, output_descriptors):
        assert i.order == j.order
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
    "reply_channel, model, input, output, output_descriptors, custom_attributes",
    [
        pytest.param(
            [],
            model_key,
            [input_key1, input_key2],
            [output_key1, output_key2],
            [output_descriptor1],
            tf_attributes,
            id="bad channel",
        ),
        pytest.param(
            b"reply channel",
            "bad model",
            [input_key1],
            [output_key2],
            [output_descriptor1],
            torch_attributes,
            id="bad model",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            ["input_key1", "input_key2"],
            [output_key1, output_key2],
            [output_descriptor1],
            tf_attributes,
            id="bad inputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [model_key],
            [output_key1, output_key2],
            [output_descriptor1],
            torch_attributes,
            id="bad input schema type",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            ["output_key1", "output_key2"],
            [output_descriptor1],
            tf_attributes,
            id="bad outputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            [model_key],
            [output_descriptor1],
            tf_attributes,
            id="bad output schema type",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            [output_key1, output_key2],
            [output_descriptor1],
            "bad attributes",
            id="bad custom attributes",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            [output_key1, output_key2],
            [output_descriptor1],
            model_key,
            id="bad custom attributes schema type",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [input_key1],
            [output_key1, output_key2],
            "bad descriptors",
            torch_attributes,
            id="bad output descriptors",
        ),
    ],
)
def test_build_request_indirect_unsuccessful(
    reply_channel, model, input, output, output_descriptors, custom_attributes
):
    with pytest.raises(ValueError):
        built_request = MessageHandler.build_request(
            reply_channel,
            model,
            input,
            output,
            output_descriptors,
            custom_attributes,
        )


@pytest.mark.parametrize(
    "reply_channel, model, input, output, output_descriptors, custom_attributes",
    [
        pytest.param(
            b"reply channel",
            model_key,
            [tensor_1, tensor_2],
            [],
            [output_descriptor2],
            torch_attributes,
        ),
        pytest.param(
            b"another reply channel",
            model,
            [tensor_1],
            [],
            [output_descriptor3],
            tf_attributes,
        ),
        pytest.param(
            b"another reply channel",
            model,
            [tensor_2],
            [],
            [output_descriptor1],
            tf_attributes,
        ),
        pytest.param(
            b"another reply channel",
            model,
            [tensor_1],
            [],
            [output_descriptor1],
            None,
        ),
    ],
)
def test_build_request_direct_successful(
    reply_channel, model, input, output, output_descriptors, custom_attributes
):
    built_request = MessageHandler.build_request(
        reply_channel,
        model,
        input,
        output,
        output_descriptors,
        custom_attributes,
    )
    assert built_request is not None
    assert built_request.replyChannel.descriptor == reply_channel
    if built_request.model.which() == "key":
        assert built_request.model.key.key == model.key
    else:
        assert built_request.model.data.data == model.data
        assert built_request.model.data.name == model.name
        assert built_request.model.data.version == model.version
    assert built_request.input.which() == "descriptors"
    assert len(built_request.input.descriptors) == len(input)
    assert len(built_request.output) == len(output)
    for i, j in zip(built_request.outputDescriptors, output_descriptors):
        assert i.order == j.order
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
    "reply_channel, model, input, output, output_descriptors, custom_attributes",
    [
        pytest.param(
            [],
            model_key,
            [tensor_3, tensor_4],
            [],
            [output_descriptor2],
            tf_attributes,
            id="bad channel",
        ),
        pytest.param(
            b"reply channel",
            "bad model",
            [tensor_4],
            [],
            [output_descriptor2],
            tf_attributes,
            id="bad model",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            ["input_key1", "input_key2"],
            [],
            [output_descriptor2],
            torch_attributes,
            id="bad inputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [],
            ["output_key1", "output_key2"],
            [output_descriptor2],
            tf_attributes,
            id="bad outputs",
        ),
        pytest.param(
            b"reply channel",
            model_key,
            [tensor_4],
            [],
            [output_descriptor2],
            "bad attributes",
            id="bad custom attributes",
        ),
        pytest.param(
            b"reply_channel",
            model_key,
            [tensor_3, tensor_4],
            [],
            ["output_descriptor2"],
            torch_attributes,
            id="bad output descriptors",
        ),
    ],
)
def test_build_request_direct_unsuccessful(
    reply_channel, model, input, output, output_descriptors, custom_attributes
):
    with pytest.raises(ValueError):
        built_request = MessageHandler.build_request(
            reply_channel,
            model,
            input,
            output,
            output_descriptors,
            custom_attributes,
        )


@pytest.mark.parametrize(
    "req",
    [
        pytest.param(tf_indirect_request, id="tf indirect"),
        pytest.param(tf_direct_request, id="tf direct"),
        pytest.param(torch_indirect_request, id="indirect"),
        pytest.param(torch_direct_request, id="direct"),
    ],
)
def test_serialize_request_successful(req):
    serialized = MessageHandler.serialize_request(req)
    assert type(serialized) == bytes

    deserialized = MessageHandler.deserialize_request(serialized)
    assert deserialized.to_dict() == req.to_dict()
