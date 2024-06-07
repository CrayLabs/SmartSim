"""This is an automatically generated stub for `request.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence, overload

from ..data.data_references_capnp import (
    ModelKey,
    ModelKeyBuilder,
    ModelKeyReader,
    TensorKey,
    TensorKeyBuilder,
    TensorKeyReader,
)
from ..tensor.tensor_capnp import (
    OutputTensorDescriptor,
    OutputTensorDescriptorBuilder,
    OutputTensorDescriptorReader,
    Tensor,
    TensorBuilder,
    TensorReader,
)
from .request_attributes.request_attributes_capnp import (
    TensorFlowRequestAttributes,
    TensorFlowRequestAttributesBuilder,
    TensorFlowRequestAttributesReader,
    TorchRequestAttributes,
    TorchRequestAttributesBuilder,
    TorchRequestAttributesReader,
)

Device = Literal["cpu", "gpu"]

class ChannelDescriptor:
    reply: bytes
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[ChannelDescriptorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> ChannelDescriptorReader: ...
    @staticmethod
    def new_message() -> ChannelDescriptorBuilder: ...
    def to_dict(self) -> dict: ...

class ChannelDescriptorReader(ChannelDescriptor):
    def as_builder(self) -> ChannelDescriptorBuilder: ...

class ChannelDescriptorBuilder(ChannelDescriptor):
    @staticmethod
    def from_dict(dictionary: dict) -> ChannelDescriptorBuilder: ...
    def copy(self) -> ChannelDescriptorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> ChannelDescriptorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class Request:
    class Model:
        modelKey: ModelKey | ModelKeyBuilder | ModelKeyReader
        modelData: bytes
        def which(self) -> Literal["modelKey", "modelData"]: ...
        def init(self, name: Literal["modelKey"]) -> ModelKey: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Request.ModelReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Request.ModelReader: ...
        @staticmethod
        def new_message() -> Request.ModelBuilder: ...
        def to_dict(self) -> dict: ...

    class ModelReader(Request.Model):
        modelKey: ModelKeyReader
        def as_builder(self) -> Request.ModelBuilder: ...

    class ModelBuilder(Request.Model):
        modelKey: ModelKey | ModelKeyBuilder | ModelKeyReader
        @staticmethod
        def from_dict(dictionary: dict) -> Request.ModelBuilder: ...
        def copy(self) -> Request.ModelBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Request.ModelReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class Device:
        deviceType: Device
        noDevice: None
        def which(self) -> Literal["deviceType", "noDevice"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Request.DeviceReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Request.DeviceReader: ...
        @staticmethod
        def new_message() -> Request.DeviceBuilder: ...
        def to_dict(self) -> dict: ...

    class DeviceReader(Request.Device):
        def as_builder(self) -> Request.DeviceBuilder: ...

    class DeviceBuilder(Request.Device):
        @staticmethod
        def from_dict(dictionary: dict) -> Request.DeviceBuilder: ...
        def copy(self) -> Request.DeviceBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Request.DeviceReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class Input:
        inputKeys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        inputData: Sequence[Tensor | TensorBuilder | TensorReader]
        def which(self) -> Literal["inputKeys", "inputData"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Request.InputReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Request.InputReader: ...
        @staticmethod
        def new_message() -> Request.InputBuilder: ...
        def to_dict(self) -> dict: ...

    class InputReader(Request.Input):
        inputKeys: Sequence[TensorKeyReader]
        inputData: Sequence[TensorReader]
        def as_builder(self) -> Request.InputBuilder: ...

    class InputBuilder(Request.Input):
        inputKeys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        inputData: Sequence[Tensor | TensorBuilder | TensorReader]
        @staticmethod
        def from_dict(dictionary: dict) -> Request.InputBuilder: ...
        def copy(self) -> Request.InputBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Request.InputReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class Output:
        outputKeys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        outputData: None
        def which(self) -> Literal["outputKeys", "outputData"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Request.OutputReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Request.OutputReader: ...
        @staticmethod
        def new_message() -> Request.OutputBuilder: ...
        def to_dict(self) -> dict: ...

    class OutputReader(Request.Output):
        outputKeys: Sequence[TensorKeyReader]
        def as_builder(self) -> Request.OutputBuilder: ...

    class OutputBuilder(Request.Output):
        outputKeys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        @staticmethod
        def from_dict(dictionary: dict) -> Request.OutputBuilder: ...
        def copy(self) -> Request.OutputBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Request.OutputReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class CustomAttributes:
        torch: (
            TorchRequestAttributes
            | TorchRequestAttributesBuilder
            | TorchRequestAttributesReader
        )
        tf: (
            TensorFlowRequestAttributes
            | TensorFlowRequestAttributesBuilder
            | TensorFlowRequestAttributesReader
        )
        none: None
        def which(self) -> Literal["torch", "tf", "none"]: ...
        @overload
        def init(self, name: Literal["torch"]) -> TorchRequestAttributes: ...
        @overload
        def init(self, name: Literal["tf"]) -> TensorFlowRequestAttributes: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Request.CustomAttributesReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Request.CustomAttributesReader: ...
        @staticmethod
        def new_message() -> Request.CustomAttributesBuilder: ...
        def to_dict(self) -> dict: ...

    class CustomAttributesReader(Request.CustomAttributes):
        torch: TorchRequestAttributesReader
        tf: TensorFlowRequestAttributesReader
        def as_builder(self) -> Request.CustomAttributesBuilder: ...

    class CustomAttributesBuilder(Request.CustomAttributes):
        torch: (
            TorchRequestAttributes
            | TorchRequestAttributesBuilder
            | TorchRequestAttributesReader
        )
        tf: (
            TensorFlowRequestAttributes
            | TensorFlowRequestAttributesBuilder
            | TensorFlowRequestAttributesReader
        )
        @staticmethod
        def from_dict(dictionary: dict) -> Request.CustomAttributesBuilder: ...
        def copy(self) -> Request.CustomAttributesBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Request.CustomAttributesReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...
    replyChannel: ChannelDescriptor | ChannelDescriptorBuilder | ChannelDescriptorReader
    model: Request.Model | Request.ModelBuilder | Request.ModelReader
    device: Request.Device | Request.DeviceBuilder | Request.DeviceReader
    input: Request.Input | Request.InputBuilder | Request.InputReader
    output: Request.Output | Request.OutputBuilder | Request.OutputReader
    outputOptions: Sequence[
        OutputTensorDescriptor
        | OutputTensorDescriptorBuilder
        | OutputTensorDescriptorReader
    ]
    customAttributes: (
        Request.CustomAttributes
        | Request.CustomAttributesBuilder
        | Request.CustomAttributesReader
    )
    @overload
    def init(self, name: Literal["replyChannel"]) -> ChannelDescriptor: ...
    @overload
    def init(self, name: Literal["model"]) -> Model: ...
    @overload
    def init(self, name: Literal["device"]) -> Device: ...
    @overload
    def init(self, name: Literal["input"]) -> Input: ...
    @overload
    def init(self, name: Literal["output"]) -> Output: ...
    @overload
    def init(self, name: Literal["customAttributes"]) -> CustomAttributes: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[RequestReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> RequestReader: ...
    @staticmethod
    def new_message() -> RequestBuilder: ...
    def to_dict(self) -> dict: ...

class RequestReader(Request):
    replyChannel: ChannelDescriptorReader
    model: Request.ModelReader
    device: Request.DeviceReader
    input: Request.InputReader
    output: Request.OutputReader
    outputOptions: Sequence[OutputTensorDescriptorReader]
    customAttributes: Request.CustomAttributesReader
    def as_builder(self) -> RequestBuilder: ...

class RequestBuilder(Request):
    replyChannel: ChannelDescriptor | ChannelDescriptorBuilder | ChannelDescriptorReader
    model: Request.Model | Request.ModelBuilder | Request.ModelReader
    device: Request.Device | Request.DeviceBuilder | Request.DeviceReader
    input: Request.Input | Request.InputBuilder | Request.InputReader
    output: Request.Output | Request.OutputBuilder | Request.OutputReader
    outputOptions: Sequence[
        OutputTensorDescriptor
        | OutputTensorDescriptorBuilder
        | OutputTensorDescriptorReader
    ]
    customAttributes: (
        Request.CustomAttributes
        | Request.CustomAttributesBuilder
        | Request.CustomAttributesReader
    )
    @staticmethod
    def from_dict(dictionary: dict) -> RequestBuilder: ...
    def copy(self) -> RequestBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> RequestReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
