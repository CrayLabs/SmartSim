import pytest
import time
from ..message_handler import MessageHandler
import torch

handler = MessageHandler()

torch1 = torch.zeros((3, 2, 5), dtype=torch.int8, layout=torch.strided)
torch2 = torch.ones((1024, 1024, 3), dtype=torch.int8, layout=torch.strided)


@pytest.mark.parametrize(
    "tensor, dtype, order, dimension",
    [
        pytest.param(torch1, "int8", "c", list(torch1.shape), id="small tensor"),
        pytest.param(torch2, "int8", "c", list(torch2.shape), id="large tensor"),
    ],
)
def test_build_tensor(tensor, dtype, order, dimension):
    start_time = time.time()
    built_tensor = handler.build_tensor(tensor, order, dtype, dimension)
    end_time = time.time()

    time_build_tensor = end_time - start_time
    formatted_time_build_tensor = "{:.6f}".format(time_build_tensor)
    print(f"Time taken to build tensor: {formatted_time_build_tensor} seconds")


built_tensor_1 = handler.build_tensor(torch1, "c", "int8", list(torch1.shape))
built_tensor_2 = handler.build_tensor(torch2, "c", "int8", list(torch2.shape))

output_tensor = handler.build_tensor(torch1, "c", "int8", list(torch1.shape))
custom_attributes = handler.build_torchcnn_request_attributes("tensor")


@pytest.mark.parametrize(
    "reply_channel, model, device, input, output, custom_attribute",
    [
        pytest.param(
            bytes("channel bytes", "utf-8"),
            bytes("model bytes", "utf-8"),
            "cpu",
            [built_tensor_1],
            [output_tensor],
            custom_attributes,
            id="small tensor",
        ),
        pytest.param(
            bytes("channel bytes", "utf-8"),
            bytes("model bytes", "utf-8"),
            "cpu",
            [built_tensor_2],
            [output_tensor],
            custom_attributes,
            id="large tensor",
        ),
    ],
)
def test_build_request(reply_channel, model, device, input, output, custom_attribute):
    start_time = time.time()
    request = handler.build_request(
        reply_channel=reply_channel,
        model=model,
        device=device,
        input=input,
        output=output,
        custom_attributes=custom_attribute,
    )
    end_time = time.time()

    time_build_request = end_time - start_time
    formatted_time_build_request = "{:.6f}".format(time_build_request)
    print(f"Time taken to build request: {formatted_time_build_request} seconds")
    assert request is not None

    # serialize time
    ser_start_time = time.time()
    serialized_request = handler.serialize_request(request)
    ser_end_time = time.time()
    time_serialize_request = ser_end_time - ser_start_time
    formatted_time_serialize_request = "{:.6f}".format(time_serialize_request)
    print(
        f"Time taken to serialize request: {formatted_time_serialize_request} seconds"
    )

    # deserialize time
    deser_start_time = time.time()
    deserialized_request = handler.deserialize_request(serialized_request)
    deser_end_time = time.time()
    time_deserialize_request = deser_end_time - deser_start_time
    formatted_time_deserialize_request = "{:.6f}".format(time_deserialize_request)
    print(
        f"Time taken to deserialize request: {formatted_time_deserialize_request} seconds"
    )


response_attributes = handler.build_torchcnn_response_attributes()
@pytest.mark.parametrize(
    "status, status_message, result, custom_attribute",
    [
        pytest.param(
            200,
            "Yay, it worked!",
            [built_tensor_1, built_tensor_2],
            None,
            id="tensor list",
        ),
        pytest.param(
            200,
            "Yay, it worked!",
            [built_tensor_1],
            response_attributes,
            id="small tensor",
        ),
    ],
)
def test_build_response(status, status_message, result, custom_attribute):
    start_time = time.time()
    response = handler.build_response(
        status=status,
        message=status_message,
        result=result,
        custom_attributes=custom_attribute,
    )
    end_time = time.time()

    time_build_response = end_time - start_time
    formatted_time_build_response = "{:.6f}".format(time_build_response)
    print(f"Time taken to build response: {formatted_time_build_response} seconds")
    assert response is not None

    # serialize time
    ser_start_time = time.time()
    serialized_response = handler.serialize_response(response)
    ser_end_time = time.time()
    time_serialize_response = ser_end_time - ser_start_time
    formatted_time_serialize_response = "{:.6f}".format(time_serialize_response)
    print(
        f"Time taken to serialize request: {formatted_time_serialize_response} seconds"
    )

    # deserialize time
    deser_start_time = time.time()
    deserialized_response = handler.deserialize_response(serialized_response)
    deser_end_time = time.time()
    time_deserialize_response = deser_end_time - deser_start_time
    formatted_time_deserialize_response = "{:.6f}".format(time_deserialize_response)
    print(
        f"Time taken to deserialize request: {formatted_time_deserialize_response} seconds"
    )