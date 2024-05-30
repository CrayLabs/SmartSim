from mli_schemas.enums.enums_capnp import *
from mli_schemas.request.request_capnp import *
from mli_schemas.request.request_attributes.request_attributes_capnp import *
from mli_schemas.response.response_capnp import *
from mli_schemas.response.response_attributes.response_attributes_capnp import *
from mli_schemas.tensor.tensor_capnp import *
import tensorflow as tf
import typing as t
import numpy as np
import torch
import time



class MessageHandler:

    # tensor info
    def build_tensor(
        self,
        data: t.Union[torch.Tensor, tf.Tensor],  # should this be a byte string instead?
        order: "Order",
        data_type: "NumericalType",
        dimensions: t.List[int],
    ) -> Tensor:
        description = TensorDescriptor.new_message()
        description.order = order
        description.dataType = data_type
        description.dimensions = dimensions
        tensor = Tensor.new_message()
        start_time = time.time()
        tensor.blob = data.numpy().tobytes()
        end_time = time.time()
        tensor.tensorDescriptor = description
        print("TIME")
        print(end_time-start_time)

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
        model: t.Union[str, t.ByteString],
        device: t.Union["Device", None],
        input: t.Union[t.List[TensorKey], Tensor],
        output: t.Union[t.List[TensorKey], Tensor],
        # maybe we send in a dict?
        custom_attributes: t.Union[
            TorchRequestAttributes, TensorflowRequestAttributes
        ],
    ) -> RequestBuilder:
        channel = ChannelDescriptor.new_message()
        channel.reply = reply_channel
        request = Request.new_message()
        request.replyChannel = channel

        if isinstance(model, str):
            request.model.modelKey = model
        else:
            request.model.modelData = model

        if device is None:
            request.device.noDevice = device
        else:
            request.device.deviceType = device

        #print(type(input)) == <class 'capnp.lib.capnp._DynamicStructBuilder'> when using direct inference, not type Tensor
        if isinstance(input, list):
            request.input.inputKeys = input
        else:
            request.input.inputData = input
    

        if isinstance(output, list):
            request.output.outputKeys = output
        else:
            request.output.outputData = output

        # if isinstance(custom_attributes, type(TorchRequestAttributes)):
        #     request.customAttributes.torchCNN = custom_attributes
        # else:
        #     request.customAttributes.tfCNN = custom_attributes

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

        if all(isinstance(item, type(Tensor)) for item in result):
            response.result.data = result
        else:
            response.result.keys = result

        if custom_attributes is None:
            response.customAttributes.none = custom_attributes
        elif isinstance(custom_attributes, type(TorchResponseAttributes)):
            response.customAttributes.torchCNN = custom_attributes
        else:
            response.customAttributes.tfCNN = custom_attributes

        return response

    def serialize_response(self, response: ResponseBuilder) -> t.ByteString:
        return response.to_bytes()

    def deserialize_response(self, response_bytes: t.ByteString) -> ResponseReader:
        bytes_message = Response.from_bytes(response_bytes)

        with bytes_message as message:
            # return a Response dataclass?
            return message

# helper functions to see if tensors can be used after being sent over
def rehydrate_torch_tensor(tensor_dict: t.Dict):
    tensor_data = np.frombuffer(
        tensor_dict["tensorData"], dtype=tensor_dict["tensorDescriptor"]["dataType"]
    )
    tensor_data = tensor_data.reshape(tensor_dict["tensorDescriptor"]["dimensions"])
    return torch.tensor(tensor_data)


def rehydrate_tf_tensor(tensor_dict: t.Dict):
    tensor_data = np.frombuffer(
        tensor_dict["tensorData"], dtype=tensor_dict["tensorDescriptor"]["dataType"]
    )
    tensor_data = tensor_data.reshape(tensor_dict["tensorDescriptor"]["dimensions"])
    return tf.constant(tensor_data)


def rehydrate_tf_tensor_list(tensor_list: t.List):
    return [rehydrate_tf_tensor(tensor) for tensor in tensor_list]


def rehydrate_torch_tensor_list(tensor_list: t.List):
    return [rehydrate_torch_tensor(tensor) for tensor in tensor_list]