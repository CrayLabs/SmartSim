"""This is an automatically generated stub for `tensor.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence

Order = Literal["c", "f"]
NumericalType = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uInt8",
    "uInt16",
    "uInt32",
    "uInt64",
    "float32",
    "float64",
]

class TensorDescriptor:
    dimensions: Sequence[int]
    order: Order
    dataType: NumericalType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorDescriptorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorDescriptorReader: ...
    @staticmethod
    def new_message() -> TensorDescriptorBuilder: ...
    def to_dict(self) -> dict: ...

class TensorDescriptorReader(TensorDescriptor):
    def as_builder(self) -> TensorDescriptorBuilder: ...

class TensorDescriptorBuilder(TensorDescriptor):
    @staticmethod
    def from_dict(dictionary: dict) -> TensorDescriptorBuilder: ...
    def copy(self) -> TensorDescriptorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorDescriptorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class Tensor:
    blob: bytes
    tensorDescriptor: (
        TensorDescriptor | TensorDescriptorBuilder | TensorDescriptorReader
    )
    def init(self, name: Literal["tensorDescriptor"]) -> TensorDescriptor: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorReader: ...
    @staticmethod
    def new_message() -> TensorBuilder: ...
    def to_dict(self) -> dict: ...

class TensorReader(Tensor):
    tensorDescriptor: TensorDescriptorReader
    def as_builder(self) -> TensorBuilder: ...

class TensorBuilder(Tensor):
    tensorDescriptor: (
        TensorDescriptor | TensorDescriptorBuilder | TensorDescriptorReader
    )
    @staticmethod
    def from_dict(dictionary: dict) -> TensorBuilder: ...
    def copy(self) -> TensorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
