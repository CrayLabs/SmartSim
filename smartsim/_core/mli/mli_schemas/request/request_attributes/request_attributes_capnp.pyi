"""This is an automatically generated stub for `request_attributes.capnp`."""

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator

from ...enums.enums_capnp import (
    TFTensorType,
    TFTensorTypeBuilder,
    TFTensorTypeReader,
    TorchTensorType,
    TorchTensorTypeBuilder,
    TorchTensorTypeReader,
)

class TorchRequestAttributes:
    tensorType: TorchTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TorchRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TorchRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TorchRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TorchRequestAttributesReader(TorchRequestAttributes):
    def as_builder(self) -> TorchRequestAttributesBuilder: ...

class TorchRequestAttributesBuilder(TorchRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TorchRequestAttributesBuilder: ...
    def copy(self) -> TorchRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TorchRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class TensorflowRequestAttributes:
    name: str
    tensorType: TFTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorflowRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorflowRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TensorflowRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TensorflowRequestAttributesReader(TensorflowRequestAttributes):
    def as_builder(self) -> TensorflowRequestAttributesBuilder: ...

class TensorflowRequestAttributesBuilder(TensorflowRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TensorflowRequestAttributesBuilder: ...
    def copy(self) -> TensorflowRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorflowRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
