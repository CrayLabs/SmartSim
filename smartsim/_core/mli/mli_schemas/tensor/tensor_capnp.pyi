"""This is an automatically generated stub for `tensor.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence, overload

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

class OutputTensorDescriptor:
    class OptionalDimension:
        dimensions: Sequence[int]
        none: None
        def which(self) -> Literal["dimensions", "none"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[OutputTensorDescriptor.OptionalDimensionReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> OutputTensorDescriptor.OptionalDimensionReader: ...
        @staticmethod
        def new_message() -> OutputTensorDescriptor.OptionalDimensionBuilder: ...
        def to_dict(self) -> dict: ...

    class OptionalDimensionReader(OutputTensorDescriptor.OptionalDimension):
        def as_builder(self) -> OutputTensorDescriptor.OptionalDimensionBuilder: ...

    class OptionalDimensionBuilder(OutputTensorDescriptor.OptionalDimension):
        @staticmethod
        def from_dict(
            dictionary: dict,
        ) -> OutputTensorDescriptor.OptionalDimensionBuilder: ...
        def copy(self) -> OutputTensorDescriptor.OptionalDimensionBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> OutputTensorDescriptor.OptionalDimensionReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class OptionalDatatype:
        dataType: NumericalType
        none: None
        def which(self) -> Literal["dataType", "none"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[OutputTensorDescriptor.OptionalDatatypeReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> OutputTensorDescriptor.OptionalDatatypeReader: ...
        @staticmethod
        def new_message() -> OutputTensorDescriptor.OptionalDatatypeBuilder: ...
        def to_dict(self) -> dict: ...

    class OptionalDatatypeReader(OutputTensorDescriptor.OptionalDatatype):
        def as_builder(self) -> OutputTensorDescriptor.OptionalDatatypeBuilder: ...

    class OptionalDatatypeBuilder(OutputTensorDescriptor.OptionalDatatype):
        @staticmethod
        def from_dict(
            dictionary: dict,
        ) -> OutputTensorDescriptor.OptionalDatatypeBuilder: ...
        def copy(self) -> OutputTensorDescriptor.OptionalDatatypeBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> OutputTensorDescriptor.OptionalDatatypeReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...
    order: Order
    optionalDimension: (
        OutputTensorDescriptor.OptionalDimension
        | OutputTensorDescriptor.OptionalDimensionBuilder
        | OutputTensorDescriptor.OptionalDimensionReader
    )
    optionalDatatype: (
        OutputTensorDescriptor.OptionalDatatype
        | OutputTensorDescriptor.OptionalDatatypeBuilder
        | OutputTensorDescriptor.OptionalDatatypeReader
    )
    @overload
    def init(self, name: Literal["optionalDimension"]) -> OptionalDimension: ...
    @overload
    def init(self, name: Literal["optionalDatatype"]) -> OptionalDatatype: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[OutputTensorDescriptorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> OutputTensorDescriptorReader: ...
    @staticmethod
    def new_message() -> OutputTensorDescriptorBuilder: ...
    def to_dict(self) -> dict: ...

class OutputTensorDescriptorReader(OutputTensorDescriptor):
    optionalDimension: OutputTensorDescriptor.OptionalDimensionReader
    optionalDatatype: OutputTensorDescriptor.OptionalDatatypeReader
    def as_builder(self) -> OutputTensorDescriptorBuilder: ...

class OutputTensorDescriptorBuilder(OutputTensorDescriptor):
    optionalDimension: (
        OutputTensorDescriptor.OptionalDimension
        | OutputTensorDescriptor.OptionalDimensionBuilder
        | OutputTensorDescriptor.OptionalDimensionReader
    )
    optionalDatatype: (
        OutputTensorDescriptor.OptionalDatatype
        | OutputTensorDescriptor.OptionalDatatypeBuilder
        | OutputTensorDescriptor.OptionalDatatypeReader
    )
    @staticmethod
    def from_dict(dictionary: dict) -> OutputTensorDescriptorBuilder: ...
    def copy(self) -> OutputTensorDescriptorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> OutputTensorDescriptorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
