"""This is an automatically generated stub for `response.capnp`."""

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence, overload

from ..tensor.tensor_capnp import (
    Tensor,
    TensorBuilder,
    TensorKey,
    TensorKeyBuilder,
    TensorKeyReader,
    TensorReader,
)
from .response_attributes.response_attributes_capnp import (
    TensorflowResponseAttributes,
    TensorflowResponseAttributesBuilder,
    TensorflowResponseAttributesReader,
    TorchResponseAttributes,
    TorchResponseAttributesBuilder,
    TorchResponseAttributesReader,
)

class Response:
    class Result:
        keys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        data: Sequence[Tensor | TensorBuilder | TensorReader]
        def which(self) -> Literal["keys", "data"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Response.ResultReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Response.ResultReader: ...
        @staticmethod
        def new_message() -> Response.ResultBuilder: ...
        def to_dict(self) -> dict: ...

    class ResultReader(Response.Result):
        keys: Sequence[TensorKeyReader]
        data: Sequence[TensorReader]
        def as_builder(self) -> Response.ResultBuilder: ...

    class ResultBuilder(Response.Result):
        keys: Sequence[TensorKey | TensorKeyBuilder | TensorKeyReader]
        data: Sequence[Tensor | TensorBuilder | TensorReader]
        @staticmethod
        def from_dict(dictionary: dict) -> Response.ResultBuilder: ...
        def copy(self) -> Response.ResultBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Response.ResultReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class CustomAttributes:
        torchCNN: (
            TorchResponseAttributes
            | TorchResponseAttributesBuilder
            | TorchResponseAttributesReader
        )
        tfCNN: (
            TensorflowResponseAttributes
            | TensorflowResponseAttributesBuilder
            | TensorflowResponseAttributesReader
        )
        none: None
        def which(self) -> Literal["torchCNN", "tfCNN", "none"]: ...
        @overload
        def init(self, name: Literal["torchCNN"]) -> TorchResponseAttributes: ...
        @overload
        def init(self, name: Literal["tfCNN"]) -> TensorflowResponseAttributes: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Response.CustomAttributesReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Response.CustomAttributesReader: ...
        @staticmethod
        def new_message() -> Response.CustomAttributesBuilder: ...
        def to_dict(self) -> dict: ...

    class CustomAttributesReader(Response.CustomAttributes):
        torchCNN: TorchResponseAttributesReader
        tfCNN: TensorflowResponseAttributesReader
        def as_builder(self) -> Response.CustomAttributesBuilder: ...

    class CustomAttributesBuilder(Response.CustomAttributes):
        torchCNN: (
            TorchResponseAttributes
            | TorchResponseAttributesBuilder
            | TorchResponseAttributesReader
        )
        tfCNN: (
            TensorflowResponseAttributes
            | TensorflowResponseAttributesBuilder
            | TensorflowResponseAttributesReader
        )
        @staticmethod
        def from_dict(dictionary: dict) -> Response.CustomAttributesBuilder: ...
        def copy(self) -> Response.CustomAttributesBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Response.CustomAttributesReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...
    status: int
    statusMessage: str
    result: Response.Result | Response.ResultBuilder | Response.ResultReader
    customAttributes: (
        Response.CustomAttributes
        | Response.CustomAttributesBuilder
        | Response.CustomAttributesReader
    )
    @overload
    def init(self, name: Literal["result"]) -> Result: ...
    @overload
    def init(self, name: Literal["customAttributes"]) -> CustomAttributes: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[ResponseReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> ResponseReader: ...
    @staticmethod
    def new_message() -> ResponseBuilder: ...
    def to_dict(self) -> dict: ...

class ResponseReader(Response):
    result: Response.ResultReader
    customAttributes: Response.CustomAttributesReader
    def as_builder(self) -> ResponseBuilder: ...

class ResponseBuilder(Response):
    result: Response.Result | Response.ResultBuilder | Response.ResultReader
    customAttributes: (
        Response.CustomAttributes
        | Response.CustomAttributesBuilder
        | Response.CustomAttributesReader
    )
    @staticmethod
    def from_dict(dictionary: dict) -> ResponseBuilder: ...
    def copy(self) -> ResponseBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> ResponseReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
