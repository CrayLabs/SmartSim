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

import numpy as np

from .mli_schemas.data import data_references_capnp
from .mli_schemas.model import model_capnp
from .mli_schemas.request import request_capnp
from .mli_schemas.request.request_attributes import request_attributes_capnp
from .mli_schemas.response import response_capnp
from .mli_schemas.response.response_attributes import response_attributes_capnp
from .mli_schemas.tensor import tensor_capnp


class MessageHandler:
    @staticmethod
    def build_tensor(
        tensor: np.ndarray[t.Any, np.dtype[t.Any]],
        order: "tensor_capnp.Order",
        data_type: "tensor_capnp.NumericalType",
        dimensions: t.List[int],
    ) -> tensor_capnp.Tensor:
        """
        Builds a Tensor message using the provided data,
        order, data type, and dimensions.

        :param tensor: Tensor to build the message around
        :param order: Order of the tensor, such as row-major (c) or column-major (f)
        :param data_type: Data type of the tensor
        :param dimensions: Dimensions of the tensor
        :raises ValueError: if building fails
        """
        try:
            description = tensor_capnp.TensorDescriptor.new_message()
            description.order = order
            description.dataType = data_type
            description.dimensions = dimensions
            built_tensor = tensor_capnp.Tensor.new_message()
            built_tensor.blob = tensor.tobytes()  # tensor channel instead?
            built_tensor.tensorDescriptor = description
        except Exception as e:
            raise ValueError(
                "Error building tensor."
            ) from e  # TODO: create custom exception

        return built_tensor

    @staticmethod
    def build_output_tensor_descriptor(
        order: "tensor_capnp.Order",
        keys: t.List["data_references_capnp.TensorKey"],
        data_type: "tensor_capnp.ReturnNumericalType",
        dimensions: t.List[int],
    ) -> tensor_capnp.OutputDescriptor:
        """
        Builds an OutputDescriptor message using the provided
        order, data type, and dimensions.

        :param order: Order of the tensor, such as row-major (c) or column-major (f)
        :param keys: List of TensorKeys to apply transorm descriptor to
        :param data_type: Tranform data type of the tensor
        :param dimensions: Transform dimensions of the tensor
        :raises ValueError: if building fails
        """
        try:
            description = tensor_capnp.OutputDescriptor.new_message()
            description.order = order
            description.optionalKeys = keys
            description.optionalDatatype = data_type
            description.optionalDimension = dimensions

        except Exception as e:
            raise ValueError("Error building output tensor descriptor.") from e

        return description

    @staticmethod
    def build_tensor_key(key: str) -> data_references_capnp.TensorKey:
        """
        Builds a new TensorKey message with the provided key.

        :param key: String to set the TensorKey
        :raises ValueError: if building fails
        """
        try:
            tensor_key = data_references_capnp.TensorKey.new_message()
            tensor_key.key = key
        except Exception as e:
            raise ValueError("Error building tensor key.") from e
        return tensor_key

    @staticmethod
    def build_model(data: bytes, name: str, version: str) -> model_capnp.Model:
        """
        Builds a new Model message with the provided data, name, and version.

        :param data: Model data
        :param name: Model name
        :param version: Model version
        :raises ValueError: if building fails
        """
        try:
            model = model_capnp.Model.new_message()
            model.data = data
            model.name = name
            model.version = version
        except Exception as e:
            raise ValueError("Error building model.") from e
        return model

    @staticmethod
    def build_model_key(key: str) -> data_references_capnp.ModelKey:
        """
        Builds a new ModelKey message with the provided key.

        :param key: String to set the ModelKey
        :raises ValueError: if building fails
        """
        try:
            model_key = data_references_capnp.ModelKey.new_message()
            model_key.key = key
        except Exception as e:
            raise ValueError("Error building model key.") from e
        return model_key

    @staticmethod
    def build_torch_request_attributes(
        tensor_type: "request_attributes_capnp.TorchTensorType",
    ) -> request_attributes_capnp.TorchRequestAttributes:
        """
        Builds a new TorchRequestAttributes message with the provided tensor type.

        :param tensor_type: Type of the tensor passed in
        :raises ValueError: if building fails
        """
        try:
            attributes = request_attributes_capnp.TorchRequestAttributes.new_message()
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building Torch request attributes.") from e
        return attributes

    @staticmethod
    def build_tf_request_attributes(
        name: str, tensor_type: "request_attributes_capnp.TFTensorType"
    ) -> request_attributes_capnp.TensorFlowRequestAttributes:
        """
        Builds a new TensorFlowRequestAttributes message with
        the provided name and tensor type.

        :param name: Name of the tensor
        :param tensor_type: Type of the tensor passed in
        :raises ValueError: if building fails
        """
        try:
            attributes = (
                request_attributes_capnp.TensorFlowRequestAttributes.new_message()
            )
            attributes.name = name
            attributes.tensorType = tensor_type
        except Exception as e:
            raise ValueError("Error building TensorFlow request attributes.") from e
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
        model: t.Union[data_references_capnp.ModelKey, model_capnp.Model],
    ) -> None:
        """
        Assigns a model to the supplied request.

        :param request: Request being built
        :param model: Model to be assigned
        :raises ValueError: if building fails
        """
        try:
            class_name = model.schema.node.displayName.split(":")[-1]  # type: ignore
            if class_name == "Model":
                request.model.data = model  # type: ignore
            elif class_name == "ModelKey":
                request.model.key = model  # type: ignore
            else:
                raise ValueError("""Invalid custom attribute class name.
                        Expected 'Model' or 'ModelKey'.""")
        except Exception as e:
            raise ValueError("Error building model portion of request.") from e

    @staticmethod
    def _assign_reply_channel(
        request: request_capnp.Request, reply_channel: bytes
    ) -> None:
        """
        Assigns a reply channel to the supplied request.

        :param request: Request being built
        :param reply_channel: Reply channel to be assigned
        :raises ValueError: if building fails
        """
        try:
            request.replyChannel.reply = reply_channel
        except Exception as e:
            raise ValueError("Error building reply channel portion of request.") from e

    @staticmethod
    def _assign_inputs(
        request: request_capnp.Request,
        inputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
    ) -> None:
        """
        Assigns inputs to the supplied request.

        :param request: Request being built
        :param inputs: Inputs to be assigned
        :raises ValueError: if building fails
        """
        try:
            if inputs:
                display_name = inputs[0].schema.node.displayName  # type: ignore
                input_class_name = display_name.split(":")[-1]
                if input_class_name == "Tensor":
                    request.input.data = inputs  # type: ignore
                elif input_class_name == "TensorKey":
                    request.input.keys = inputs  # type: ignore
                else:
                    raise ValueError(
                        "Invalid input class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error building inputs portion of request.") from e

    @staticmethod
    def _assign_outputs(
        request: request_capnp.Request,
        outputs: t.List[data_references_capnp.TensorKey],
    ) -> None:
        """
        Assigns outputs to the supplied request.

        :param request: Request being built
        :param outputs: Outputs to be assigned
        :raises ValueError: if building fails
        """
        try:
            request.output = outputs

        except Exception as e:
            raise ValueError("Error building outputs portion of request.") from e

    @staticmethod
    def _assign_output_descriptors(
        request: request_capnp.Request,
        output_descriptors: t.List[tensor_capnp.OutputDescriptor],
    ) -> None:
        """
        Assigns a list of output tensor descriptors to the supplied request.

        :param request: Request being built
        :param output_descriptors: Output descriptors to be assigned
        :raises ValueError: if building fails
        """
        try:
            request.outputDescriptors = output_descriptors
        except Exception as e:
            raise ValueError(
                "Error building the output descriptors portion of request."
            ) from e

    @staticmethod
    def _assign_custom_request_attributes(
        request: request_capnp.Request,
        custom_attrs: t.Union[
            request_attributes_capnp.TorchRequestAttributes,
            request_attributes_capnp.TensorFlowRequestAttributes,
            None,
        ],
    ) -> None:
        """
        Assigns request attributes to the supplied request.

        :param request: Request being built
        :param custom_attrs: Custom attributes to be assigned
        :raises ValueError: if building fails
        """
        try:
            if custom_attrs is None:
                request.customAttributes.none = custom_attrs
            else:
                custom_attribute_class_name = (
                    custom_attrs.schema.node.displayName.split(":")[-1]  # type: ignore
                )
                if custom_attribute_class_name == "TorchRequestAttributes":
                    request.customAttributes.torch = custom_attrs  # type: ignore
                elif custom_attribute_class_name == "TensorFlowRequestAttributes":
                    request.customAttributes.tf = custom_attrs  # type: ignore
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
        reply_channel: bytes,
        model: t.Union[data_references_capnp.ModelKey, model_capnp.Model],
        inputs: t.Union[
            t.List[data_references_capnp.TensorKey], t.List[tensor_capnp.Tensor]
        ],
        outputs: t.List[data_references_capnp.TensorKey],
        output_descriptors: t.List[tensor_capnp.OutputDescriptor],
        custom_attributes: t.Union[
            request_attributes_capnp.TorchRequestAttributes,
            request_attributes_capnp.TensorFlowRequestAttributes,
            None,
        ],
    ) -> request_capnp.Request:
        """
        Builds the request message.

        :param reply_channel: Reply channel to be assigned to request
        :param model: Model to be assigned to request
        :param inputs: Inputs to be assigned to request
        :param outputs: Outputs to be assigned to request
        :param output_descriptors: Output descriptors to be assigned to request
        :param custom_attributes: Custom attributes to be assigned to request
        """
        request = request_capnp.Request.new_message()
        MessageHandler._assign_reply_channel(request, reply_channel)
        MessageHandler._assign_model(request, model)
        MessageHandler._assign_inputs(request, inputs)
        MessageHandler._assign_outputs(request, outputs)
        MessageHandler._assign_output_descriptors(request, output_descriptors)
        MessageHandler._assign_custom_request_attributes(request, custom_attributes)
        return request

    @staticmethod
    def serialize_request(request: request_capnp.RequestBuilder) -> bytes:
        """
        Serializes a built request message.

        :param request: Request to be serialized
        """
        return request.to_bytes()

    @staticmethod
    def deserialize_request(request_bytes: bytes) -> request_capnp.Request:
        """
        Deserializes a serialized request message.

        :param request_bytes: Bytes to be deserialized into a Request
        """
        bytes_message = request_capnp.Request.from_bytes(
            request_bytes, traversal_limit_in_words=2**63
        )

        with bytes_message as message:
            return message

    @staticmethod
    def _assign_status(
        response: response_capnp.Response, status: "response_capnp.Status"
    ) -> None:
        """
        Assigns a status to the supplied response.

        :param response: Response being built
        :param status: Status to be assigned
        :raises ValueError: if building fails
        """
        try:
            response.status = status
        except Exception as e:
            raise ValueError("Error assigning status to response.") from e

    @staticmethod
    def _assign_message(response: response_capnp.Response, message: str) -> None:
        """
        Assigns a message to the supplied response.

        :param response: Response being built
        :param message: Message to be assigned
        :raises ValueError: if building fails
        """
        try:
            response.message = message
        except Exception as e:
            raise ValueError("Error assigning message to response.") from e

    @staticmethod
    def _assign_result(
        response: response_capnp.Response,
        result: t.Union[
            t.List[tensor_capnp.Tensor], t.List[data_references_capnp.TensorKey]
        ],
    ) -> None:
        """
        Assigns a result to the supplied response.

        :param response: Response being built
        :param result: Result to be assigned
        :raises ValueError: if building fails
        """
        try:
            if result:
                first_result = result[0]
                display_name = first_result.schema.node.displayName  # type: ignore
                result_class_name = display_name.split(":")[-1]
                if result_class_name == "Tensor":
                    response.result.data = result  # type: ignore
                elif result_class_name == "TensorKey":
                    response.result.keys = result  # type: ignore
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'Tensor' or 'TensorKey'.""")
        except Exception as e:
            raise ValueError("Error assigning result to response.") from e

    @staticmethod
    def _assign_custom_response_attributes(
        response: response_capnp.Response,
        custom_attrs: t.Union[
            response_attributes_capnp.TorchResponseAttributes,
            response_attributes_capnp.TensorFlowResponseAttributes,
            None,
        ],
    ) -> None:
        """
        Assigns custom attributes to the supplied response.

        :param response: Response being built
        :param custom_attrs: Custom attributes to be assigned
        :raises ValueError: if building fails
        """
        try:
            if custom_attrs is None:
                response.customAttributes.none = custom_attrs
            else:
                custom_attribute_class_name = (
                    custom_attrs.schema.node.displayName.split(":")[-1]  # type: ignore
                )
                if custom_attribute_class_name == "TorchResponseAttributes":
                    response.customAttributes.torch = custom_attrs  # type: ignore
                elif custom_attribute_class_name == "TensorFlowResponseAttributes":
                    response.customAttributes.tf = custom_attrs  # type: ignore
                else:
                    raise ValueError("""Invalid custom attribute class name.
                        Expected 'TensorFlowResponseAttributes' or
                        'TorchResponseAttributes'.""")
        except Exception as e:
            raise ValueError("Error assigning custom attributes to response.") from e

    @staticmethod
    def build_response(
        status: "response_capnp.Status",
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

        :param status: Status to be assigned to response
        :param message: Message to be assigned to response
        :param result: Result to be assigned to response
        :param custom_attributes: Custom attributes to be assigned to response
        """
        response = response_capnp.Response.new_message()
        MessageHandler._assign_status(response, status)
        MessageHandler._assign_message(response, message)
        MessageHandler._assign_result(response, result)
        MessageHandler._assign_custom_response_attributes(response, custom_attributes)
        return response

    @staticmethod
    def serialize_response(response: response_capnp.ResponseBuilder) -> bytes:
        """
        Serializes a built response message.
        """
        return response.to_bytes()

    @staticmethod
    def deserialize_response(response_bytes: bytes) -> response_capnp.Response:
        """
        Deserializes a serialized response message.
        """
        bytes_message = response_capnp.Response.from_bytes(
            response_bytes, traversal_limit_in_words=2**63
        )

        with bytes_message as message:
            return message
