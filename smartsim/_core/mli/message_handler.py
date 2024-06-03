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
import typing as t

import capnp  # pylint: disable=unused-import
import tensorflow as tf
import torch

from .mli_schemas.request import request_capnp  # pylint: disable=no-name-in-module
from .mli_schemas.request.request_attributes import (
    request_attributes_capnp,  # pylint: disable=no-name-in-module
)
from .mli_schemas.response import response_capnp  # pylint: disable=no-name-in-module
from .mli_schemas.response.response_attributes import (
    response_attributes_capnp,  # pylint: disable=no-name-in-module
)
from .mli_schemas.tensor import tensor_capnp  # pylint: disable=no-name-in-module


class MessageHandler:
    @staticmethod
    def build_tensor(
        tensor: t.Union[torch.Tensor, tf.Tensor],
        order: str,
        data_type: str,
        dimensions: t.List[int],
    ) -> tensor_capnp.Tensor:
        """
        Builds a tensor using the provided data, order, data type, and dimensions.
        """
        try:
            description = tensor_capnp.TensorDescriptor.new_message()
            description.order = order
            description.dataType = data_type
            description.dimensions = dimensions
            built_tensor = tensor_capnp.Tensor.new_message()
            built_tensor.blob = tensor.numpy().tobytes()  # tensor channel instead?
            built_tensor.tensorDescriptor = description
        except Exception as e:
            raise ValueError(
                "Error building tensor."
            ) from e  # TODO: create custom exception

        return built_tensor

    @staticmethod
    def build_tensor_key(key: str) -> tensor_capnp.TensorKey:
        try:
            tensor_key = tensor_capnp.TensorKey.new_message()
            tensor_key.key = key
        except Exception as e:
            raise ValueError("Error building tensor key.") from e
        return tensor_key

    @staticmethod
    def build_model_key(key: str) -> request_capnp.ModelKey:
        try:
            model_key = request_capnp.ModelKey.new_message()
            model_key.key = key
        except Exception as e:
            raise ValueError("Error building model key.") from e
        return model_key

    @staticmethod
    def build_torchcnn_request_attributes(
        tensor_type: str,
    ) -> request_attributes_capnp.TorchRequestAttributes:
        try:
            attributes = request_attributes_capnp.TorchRequestAttributes.new_message()
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building torchcnn request attributes.") from e
        return attributes

    @staticmethod
    def build_tfcnn_request_attributes(
        name: str, tensor_type: str
    ) -> request_attributes_capnp.TensorflowRequestAttributes:
        try:
            attributes = (
                request_attributes_capnp.TensorflowRequestAttributes.new_message()
            )
            attributes.name = name
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building tfcnn request attributes.") from e
        return attributes

    @staticmethod
    def build_torchcnn_response_attributes() -> (
        response_attributes_capnp.TorchResponseAttributes
    ):
        return response_attributes_capnp.TorchResponseAttributes.new_message()

    @staticmethod
    def build_tfcnn_response_attributes() -> (
        response_attributes_capnp.TensorflowResponseAttributes
    ):
        return response_attributes_capnp.TensorflowResponseAttributes.new_message()

    @staticmethod
    def build_request(
        reply_channel: t.ByteString,
        model: t.Union[request_capnp.ModelKey, t.ByteString],
        device: t.Union[str, None],
        inputs: t.Union[t.List[tensor_capnp.TensorKey], t.List[tensor_capnp.Tensor]],
        outputs: t.Union[t.List[tensor_capnp.TensorKey], t.List[tensor_capnp.Tensor]],
        custom_attributes: t.Union[
            request_attributes_capnp.TorchRequestAttributes,
            request_attributes_capnp.TensorflowRequestAttributes,
            None,
        ],
    ) -> request_capnp.Request:
        try:
            channel = request_capnp.ChannelDescriptor.new_message()
            channel.reply = reply_channel
            request = request_capnp.Request.new_message()
            request.replyChannel = channel
        except Exception as e:
            raise ValueError("Error building reply channel portion of request.") from e

        try:
            if isinstance(model, bytes):
                request.model.modelData = model
            else:
                request.model.modelKey = model
        except Exception as e:
            raise ValueError("Error building model portion of request.") from e

        try:
            if device is None:
                request.device.noDevice = device
            else:
                request.device.deviceType = device
        except Exception as e:
            raise ValueError("Error building device portion request.") from e

        try:
            if inputs:
                first_input = inputs[0]
                input_class_name = first_input.schema.node.displayName.split(":")[-1]
                if input_class_name == "Tensor":
                    request.input.inputData = inputs
                elif input_class_name == "TensorKey":
                    request.input.inputKeys = inputs
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'Tensor' or 'TensorKey'.""")
        except Exception as e:
            raise ValueError("Error building inputs portion ofrequest.") from e

        try:
            if outputs:
                first_output = outputs[0]
                output_class_name = first_output.schema.node.displayName.split(":")[-1]
                if output_class_name == "Tensor":
                    request.output.outputData = outputs
                elif output_class_name == "TensorKey":
                    request.output.outputKeys = outputs
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'Tensor' or 'TensorKey'.""")
        except Exception as e:
            raise ValueError("Error building outputs portion of request.") from e

        try:
            if custom_attributes is None:
                request.customAttributes.none = custom_attributes
            else:
                custom_attribute_class_name = (
                    custom_attributes.schema.node.displayName.split(":")[-1]
                )
                if custom_attribute_class_name == "TorchRequestAttributes":
                    request.customAttributes.torchCNN = custom_attributes
                elif custom_attribute_class_name == "TensorflowRequestAttributes":
                    request.customAttributes.tfCNN = custom_attributes
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'TensorflowRequestAttributes' or
                        'TorchRequestAttributes'.""")
        except Exception as e:
            raise ValueError(
                "Error building custom attributes portion of request."
            ) from e

        return request

    @staticmethod
    def serialize_request(request: request_capnp.Request) -> t.ByteString:
        return request.to_bytes()

    @staticmethod
    def deserialize_request(request_bytes: t.ByteString) -> request_capnp.Request:
        bytes_message = request_capnp.Request.from_bytes(request_bytes)

        with bytes_message as message:
            # return a Request dataclass?
            return message

    @staticmethod
    def build_response(
        status: int,
        message: str,
        result: t.Union[t.List[tensor_capnp.Tensor], t.List[tensor_capnp.TensorKey]],
        custom_attributes: t.Union[
            response_attributes_capnp.TorchResponseAttributes,
            response_attributes_capnp.TensorflowResponseAttributes,
            None,
        ],
    ) -> response_capnp.Response:
        try:
            response = response_capnp.Response.new_message()
            response.status = status
            response.statusMessage = message
        except Exception as e:
            raise ValueError("Error building response.") from e

        try:
            if result:
                first_result = result[0]
                result_class_name = first_result.schema.node.displayName.split(":")[-1]
                if result_class_name == "Tensor":
                    response.result.data = result
                elif result_class_name == "TensorKey":
                    response.result.keys = result
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'Tensor' or 'TensorKey'.""")
        except Exception as e:
            raise ValueError("Error building result portion of response.") from e

        try:
            if custom_attributes is None:
                response.customAttributes.none = custom_attributes
            else:
                custom_attribute_class_name = (
                    custom_attributes.schema.node.displayName.split(":")[-1]
                )
                if custom_attribute_class_name == "TorchResponseAttributes":
                    response.customAttributes.torchCNN = custom_attributes
                elif custom_attribute_class_name == "TensorflowResponseAttributes":
                    response.customAttributes.tfCNN = custom_attributes
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'TensorflowResponseAttributes' or 
                        'TorchResponseAttributes'.""")
        except Exception as e:
            raise ValueError(
                "Error building custom attributes portion of response."
            ) from e

        return response

    @staticmethod
    def serialize_response(response: response_capnp.Response) -> t.ByteString:
        return response.to_bytes()

    @staticmethod
    def deserialize_response(response_bytes: t.ByteString) -> response_capnp.Response:
        bytes_message = response_capnp.Response.from_bytes(response_bytes)

        with bytes_message as message:
            # return a Response dataclass?
            return message
