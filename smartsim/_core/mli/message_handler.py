from .mli_schemas.enums.enums_capnp import *
from .mli_schemas.request.request_capnp import *
from .mli_schemas.request.request_attributes.request_attributes_capnp import *
from .mli_schemas.response.response_capnp import *
from .mli_schemas.response.response_attributes.response_attributes_capnp import *
from .mli_schemas.tensor.tensor_capnp import *
import tensorflow as tf
import typing as t
import numpy as np
import torch


class MessageHandler:

    # tensor info
    def build_tensor(
        self,
        data: t.Union[torch.Tensor, tf.Tensor],
        order: "Order",
        data_type: "NumericalType",
        dimensions: t.List[int],
    ) -> Tensor:
        description = TensorDescriptor.new_message()
        description.order = order
        description.dataType = data_type
        description.dimensions = dimensions
        tensor = Tensor.new_message()
        tensor.blob = data.numpy().tobytes()
        tensor.tensorDescriptor = description

        return tensor

    # attributes
    def build_tensor_key(self, name: str) -> TensorKey:
        tensor_key = TensorKey.new_message()
        tensor_key.key = name
        return tensor_key

    def build_torchcnn_request_attributes(
        self, tensor_type: "TorchTensorType"
    ) -> TorchRequestAttributes:
        attributes = TorchRequestAttributes.new_message()
        attributes.tensorType = tensor_type
        return attributes

    def build_tfcnn_request_attributes(
        self, name: str, tensor_type: "TFTensorType"
    ) -> TensorflowRequestAttributes:
        attributes = TensorflowRequestAttributes.new_message()
        attributes.name = name
        attributes.tensorType = tensor_type
        return attributes

    # response attributes are currently empty
    def build_torchcnn_response_attributes(self) -> TorchResponseAttributes:
        return TorchResponseAttributes.new_message()

    def build_tfcnn_response_attributes(self) -> TensorflowResponseAttributes:
        return TensorflowResponseAttributes.new_message()

    # request
    def build_request(
        self,
        reply_channel: t.ByteString,
        model: t.Union[ModelKey, t.ByteString],
        device: t.Union["Device", None],
        input: t.Union[t.List[TensorKey], t.List[Tensor]],
        output: t.Union[t.List[TensorKey], t.List[Tensor]],
        custom_attributes: t.Union[TorchRequestAttributes, TensorflowRequestAttributes],
    ) -> RequestBuilder:
        channel = ChannelDescriptor.new_message()
        channel.reply = reply_channel
        request = Request.new_message()
        request.replyChannel = channel

        if isinstance(model, t.ByteString):
            request.model.modelData = model
        else:
            request.model.modelKey = model

        if device is None:
            request.device.noDevice = device
        else:
            request.device.deviceType = device


        try:
            if input:
                first_input = input[0]
                input_class_name = first_input.schema.node.displayName.split(":")[-1]
                print(input_class_name)
                if input_class_name == "Tensor":
                    request.input.inputData = input
                elif input_class_name == "TensorKey":
                    request.input.inputKeys = input
                else:
                    raise ValueError(
                        "Invalid custom attribute class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error accessing custom attribute information.") from e

        try:
            if output:
                first_output = output[0]
                output_class_name = first_output.schema.node.displayName.split(":")[-1]
                print(output_class_name)
                if output_class_name == "Tensor":
                    request.output.outputData = output
                elif output_class_name == "TensorKey":
                    request.output.outputKeys = output
                else:
                    raise ValueError(
                        "Invalid custom attribute class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error accessing custom attribute information.") from e

        try:
            custom_attribute_class_name = (
                custom_attributes.schema.node.displayName.split(":")[-1]
            )
            print(custom_attribute_class_name)
            if custom_attribute_class_name == "TorchRequestAttributes":
                request.customAttributes.torchCNN = custom_attributes
            elif custom_attribute_class_name == "TensorflowRequestAttributes":
                request.customAttributes.tfCNN = custom_attributes
            else:
                raise ValueError(
                    "Invalid custom attribute class name. Expected 'TensorflowRequestAttributes' or 'TorchRequestAttributes'."
                )
        except Exception as e:
            raise ValueError("Error accessing custom attribute information.") from e

        return request

    def serialize_request(self, request: RequestBuilder) -> t.ByteString:
        return request.to_bytes()

    def deserialize_request(self, request_bytes: t.ByteString) -> RequestReader:
        bytes_message = Request.from_bytes(request_bytes)

        with bytes_message as message:
            # return a Request dataclass?
            return message

    # response
    def build_response(
        self,
        status: int,
        message: str,
        result: t.Union[t.List[Tensor], t.List[TensorKey]],
        custom_attributes: t.Union[
            TorchResponseAttributes, TensorflowResponseAttributes, None
        ],
    ) -> ResponseBuilder:
        response = Response.new_message()
        response.status = status
        response.statusMessage = message

        try:
            if result:
                first_result = result[0]
                result_class_name = first_result.schema.node.displayName.split(":")[-1]
                print(result_class_name)
                if result_class_name == "Tensor":
                    response.result.data = result
                elif result_class_name == "TensorKey":
                    response.result.keys = result
                else:
                    raise ValueError(
                        "Invalid custom attribute class name. Expected 'Tensor' or 'TensorKey'."
                    )
        except Exception as e:
            raise ValueError("Error accessing custom attribute information.") from e

        if custom_attributes is None:
            response.customAttributes.none = custom_attributes
        else:
            try:
                custom_attribute_class_name = (
                    custom_attributes.schema.node.displayName.split(":")[-1]
                )
                print(custom_attribute_class_name)
                if custom_attribute_class_name == "TorchResponseAttributes":
                    response.customAttributes.torchCNN = custom_attributes
                elif custom_attribute_class_name == "TensorflowResponseAttributes":
                    response.customAttributes.tfCNN = custom_attributes
                else:
                    raise ValueError(
                        "Invalid custom attribute class name. Expected 'TensorflowRequestAttributes' or 'TorchRequestAttributes'."
                    )
            except Exception as e:
                raise ValueError("Error accessing custom attribute information.") from e

        return response

    def serialize_response(self, response: ResponseBuilder) -> t.ByteString:
        return response.to_bytes()

    def deserialize_response(self, response_bytes: t.ByteString) -> ResponseReader:
        bytes_message = Response.from_bytes(response_bytes)

        with bytes_message as message:
            # return a Response dataclass?
            return message


# helper functions to see if tensors can be used after being sent over
def rehydrate_torch_tensor(tensor: Tensor):
    tensor_data = np.frombuffer(
        tensor.blob, dtype=tensor.tensorDescriptor.dataType
    )
    tensor_data = tensor_data.reshape(tensor.tensorDescriptor.dimensions)
    return torch.tensor(tensor_data)


def rehydrate_tf_tensor(tensor: Tensor):
    tensor_data = np.frombuffer(
        tensor.blob, dtype=tensor.tensorDescriptor.dataType
    )
    tensor_data = tensor_data.reshape(tensor.tensorDescriptor.dimensions)
    return tf.constant(tensor_data)


def rehydrate_tf_tensor_list(tensor_list: t.List):
    return [rehydrate_tf_tensor(tensor) for tensor in tensor_list]


def rehydrate_torch_tensor_list(tensor_list: t.List):
    return [rehydrate_torch_tensor(tensor) for tensor in tensor_list]
