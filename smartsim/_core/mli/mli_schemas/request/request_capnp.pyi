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
from ..model.model_capnp import Model, ModelBuilder, ModelReader
from ..tensor.tensor_capnp import (
    OutputDescriptor,
    OutputDescriptorBuilder,
    OutputDescriptorReader,
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
        modelData: Model | ModelBuilder | ModelReader
        def which(self) -> Literal["modelKey", "modelData"]: ...
        @overload
        def init(self, name: Literal["modelKey"]) -> ModelKey: ...
        @overload
        def init(self, name: Literal["modelData"]) -> Model: ...
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
        modelData: ModelReader
        def as_builder(self) -> Request.ModelBuilder: ...

    class ModelBuilder(Request.Model):
        modelKey: ModelKey | ModelKeyBuilder | ModelKeyReader
        modelData: Model | ModelBuilder | ModelReader
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
    input: Request.Input | Request.InputBuilder | Request.InputReader
    output: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
    outputDescriptors: Sequence[
        OutputDescriptor | OutputDescriptorBuilder | OutputDescriptorReader
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
    def init(self, name: Literal["input"]) -> Input: ...
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
    input: Request.InputReader
    output: Sequence[TensorKeyReader]
    outputDescriptors: Sequence[OutputDescriptorReader]
    customAttributes: Request.CustomAttributesReader
    def as_builder(self) -> RequestBuilder: ...

class RequestBuilder(Request):
    replyChannel: ChannelDescriptor | ChannelDescriptorBuilder | ChannelDescriptorReader
    model: Request.Model | Request.ModelBuilder | Request.ModelReader
    input: Request.Input | Request.InputBuilder | Request.InputReader
    output: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
    outputDescriptors: Sequence[
        OutputDescriptor | OutputDescriptorBuilder | OutputDescriptorReader
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
