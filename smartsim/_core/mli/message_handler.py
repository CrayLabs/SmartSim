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

import capnp  # type: ignore # pylint: disable=unused-import
import tensorflow as tf
import torch

from .mli_schemas.data import (  # type: ignore # pylint: disable=no-name-in-module
    data_references_capnp,
)
from .mli_schemas.request import (  # type: ignore # pylint: disable=no-name-in-module
    request_capnp,
)
from .mli_schemas.request.request_attributes import (  # type: ignore # pylint: disable=no-name-in-module
    request_attributes_capnp,
)
from .mli_schemas.response import (  # type: ignore # pylint: disable=no-name-in-module
    response_capnp,
)
from .mli_schemas.response.response_attributes import (  # type: ignore # pylint: disable=no-name-in-module
    response_attributes_capnp,
)
from .mli_schemas.tensor import (  # type: ignore # pylint: disable=no-name-in-module
    tensor_capnp,
)


class MessageHandler:
    @staticmethod
    def build_tensor(
        tensor: t.Union[torch.Tensor, tf.Tensor],
        order: str,
        data_type: str,
        dimensions: t.List[int],
    ) -> tensor_capnp.Tensor:
        """
        Builds a Tensor message using the provided data,
        order, data type, and dimensions.
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
    def build_tensor_key(key: str) -> data_references_capnp.TensorKey:
        """
        Builds a new TensorKey message with the provided key.
        """
        try:
            tensor_key = data_references_capnp.TensorKey.new_message()
            tensor_key.key = key
        except Exception as e:
            raise ValueError("Error building tensor key.") from e
        return tensor_key

    @staticmethod
    def build_model_key(key: str) -> data_references_capnp.ModelKey:
        """
        Builds a new ModelKey message with the provided key.
        """
        try:
            model_key = data_references_capnp.ModelKey.new_message()
            model_key.key = key
        except Exception as e:
            raise ValueError("Error building model key.") from e
        return model_key

    @staticmethod
    def build_torch_request_attributes(
        tensor_type: str,
    ) -> request_attributes_capnp.TorchRequestAttributes:
        """
        Builds a new TorchRequestAttributes message with the provided tensor type.
        """
        try:
            attributes = request_attributes_capnp.TorchRequestAttributes.new_message()
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building torch request attributes.") from e
        return attributes

    @staticmethod
    def build_tf_request_attributes(
        name: str, tensor_type: str
    ) -> request_attributes_capnp.TensorFlowRequestAttributes:
        """
        Builds a new TensorFlowRequestAttributes message with
        the provided name and tensor type.
        """
        try:
            attributes = (
                request_attributes_capnp.TensorFlowRequestAttributes.new_message()
            )
            attributes.name = name
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building tf request attributes.") from e
        return attributes

    @staticmethod
    def build_torch_response_attributes() -> (
        response_attributes_capnp.TorchResponseAttributes
    ):
        """
        Builds a new TorchResponseAttributes message.
        """
        return response_attributes_capnp.TorchResponseAttributes.new_message()

    @staticmethod
    def build_tf_response_attributes() -> (
        response_attributes_capnp.TensorFlowResponseAttributes
    ):
        """
        Builds a new TensorFlowResponseAttributes message.
        """
        return response_attributes_capnp.TensorFlowResponseAttributes.new_message()

    @staticmethod
    def _assign_model(
        request: request_capnp.Request,
        model: t.Union[data_references_capnp.ModelKey, t.ByteString],
    ) -> None:
        """
        Assigns a model to the supplied request.
        """
        try:
            if isinstance(model, bytes):
                request.model.modelData = model
            else:
                request.model.modelKey = model
        except Exception as e:
            raise ValueError("Error building model portion of request.") from e

    @staticmethod
    def _assign_reply_channel(
        request: request_capnp.Request, reply_channel: t.ByteString
    ) -> None:
        """
        Assigns a reply channel to the supplied request.
        """
        try:
            request.replyChannel.reply = reply_channel
        except Exception as e:
            raise ValueError("Error building reply channel portion of request.") from e

    @staticmethod
    def _assign_device(
        request: request_capnp.Request, device: t.Union[str, None]
    ) -> None:
        """
        Assigns a device to the supplied request.
        """
        try:
            if device is None:
                request.device.noDevice = device
            else:
                request.device.deviceType = device
        except Exception as e:
            raise ValueError("Error building device portion of request.") from e

    @staticmethod
    def _assign_inputs(
        request: request_capnp.Request,
        inputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
    ) -> None:
        """
        Assigns inputs to the supplied request.
        """
        try:
            if inputs:
                input_class_name = inputs[0].schema.node.displayName.split(":")[-1]
                if input_class_name == "Tensor":
                    request.input.inputData = inputs
                elif input_class_name == "TensorKey":
                    request.input.inputKeys = inputs
                else:
                    raise ValueError(
                        "Invalid input class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error building inputs portion of request.") from e

    @staticmethod
    def _assign_outputs(
        request: request_capnp.Request,
        outputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
    ) -> None:
        """
        Assigns outputs to the supplied request.
        """
        try:
            if outputs:
                output_class_name = outputs[0].schema.node.displayName.split(":")[-1]
                if output_class_name == "Tensor":
                    request.output.outputData = outputs
                elif output_class_name == "TensorKey":
                    request.output.outputKeys = outputs
                else:
                    raise ValueError(
                        "Invalid output class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error building outputs portion of request.") from e

    @staticmethod
    def _assign_custom_request_attributes(
        request: request_capnp.Request,
        custom_attributes: t.Union[
            request_attributes_capnp.TorchRequestAttributes,
            request_attributes_capnp.TensorFlowRequestAttributes,
            None,
        ],
    ) -> None:
        """
        Assigns request attributes to the supplied request.
        """
        try:
            if custom_attributes is None:
                request.customAttributes.none = custom_attributes
            else:
                custom_attribute_class_name = (
                    custom_attributes.schema.node.displayName.split(":")[-1]
                )
                if custom_attribute_class_name == "TorchRequestAttributes":
                    request.customAttributes.torch = custom_attributes
                elif custom_attribute_class_name == "TensorFlowRequestAttributes":
                    request.customAttributes.tf = custom_attributes
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'TensorFlowRequestAttributes' or
                        'TorchRequestAttributes'.""")
        except Exception as e:
            raise ValueError(
                "Error building custom attributes portion of request."
            ) from e

    @staticmethod
    def build_request(
        reply_channel: t.ByteString,
        model: t.Union[data_references_capnp.ModelKey, t.ByteString],
        device: t.Union[str, None],
        inputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
        outputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
        custom_attributes: t.Union[
            request_attributes_capnp.TorchRequestAttributes,
            request_attributes_capnp.TensorFlowRequestAttributes,
            None,
        ],
    ) -> request_capnp.Request:
        """
        Builds the request message.
        """
        request = request_capnp.Request.new_message()
        MessageHandler._assign_reply_channel(request, reply_channel)
        MessageHandler._assign_model(request, model)
        MessageHandler._assign_device(request, device)
        MessageHandler._assign_inputs(request, inputs)
        MessageHandler._assign_outputs(request, outputs)
        MessageHandler._assign_custom_request_attributes(request, custom_attributes)
        return request

    @staticmethod
    def serialize_request(request: request_capnp.Request) -> t.ByteString:
        """
        Serializes a built request message.
        """
        return request.to_bytes()  # type: ignore

    @staticmethod
    def deserialize_request(request_bytes: t.ByteString) -> request_capnp.Request:
        """
        Deserializes a serialized request message.
        """
        bytes_message = request_capnp.Request.from_bytes(request_bytes)

        with bytes_message as message:
            # return a Request dataclass?
            return message

    @staticmethod
    def _assign_status(response: response_capnp.Response, status: int) -> None:
        """
        Assigns a status to the supplied response.
        """
        try:
            response.status = status
        except Exception as e:
            raise ValueError("Error assigning status to response.") from e

    @staticmethod
    def _assign_status_message(response: response_capnp.Response, message: str) -> None:
        """
        Assigns a status message to the supplied response.
        """
        try:
            response.statusMessage = message
        except Exception as e:
            raise ValueError("Error assigning status message to response.") from e

    @staticmethod
    def _assign_result(
        response: response_capnp.Response,
        result: t.Union[
            t.List[tensor_capnp.Tensor], t.List[data_references_capnp.TensorKey]
        ],
    ) -> None:
        """
        Assigns a result to the supplied response.
        """
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
            raise ValueError("Error assigning result to response.") from e

    @staticmethod
    def _assign_custom_response_attributes(
        response: response_capnp.Response,
        custom_attributes: t.Union[
            response_attributes_capnp.TorchResponseAttributes,
            response_attributes_capnp.TensorFlowResponseAttributes,
            None,
        ],
    ) -> None:
        """
        Assigns custom attributes to the supplied response.
        """
        try:
            if custom_attributes is None:
                response.customAttributes.none = custom_attributes
            else:
                custom_attribute_class_name = (
                    custom_attributes.schema.node.displayName.split(":")[-1]
                )
                if custom_attribute_class_name == "TorchResponseAttributes":
                    response.customAttributes.torch = custom_attributes
                elif custom_attribute_class_name == "TensorFlowResponseAttributes":
                    response.customAttributes.tf = custom_attributes
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'TensorFlowResponseAttributes' or 
                        'TorchResponseAttributes'.""")
        except Exception as e:
            raise ValueError("Error assigning custom attributes to response.") from e

    @staticmethod
    def build_response(
        status: int,
        message: str,
        result: t.Union[
            t.List[tensor_capnp.Tensor], t.List[data_references_capnp.TensorKey]
        ],
        custom_attributes: t.Union[
            response_attributes_capnp.TorchResponseAttributes,
            response_attributes_capnp.TensorFlowResponseAttributes,
            None,
        ],
    ) -> response_capnp.Response:
        """
        Builds the response message.
        """
        response = response_capnp.Response.new_message()
        MessageHandler._assign_status(response, status)
        MessageHandler._assign_status_message(response, message)
        MessageHandler._assign_result(response, result)
        MessageHandler._assign_custom_response_attributes(response, custom_attributes)
        return response

    @staticmethod
    def serialize_response(response: response_capnp.Response) -> t.ByteString:
        """
        Serializes a built response message.
        """
        return response.to_bytes()  # type: ignore

    @staticmethod
    def deserialize_response(response_bytes: t.ByteString) -> response_capnp.Response:
        """
        Deserializes a serialized response message.
        """
        bytes_message = response_capnp.Response.from_bytes(response_bytes)

        with bytes_message as message:
            # return a Response dataclass?
            return message
