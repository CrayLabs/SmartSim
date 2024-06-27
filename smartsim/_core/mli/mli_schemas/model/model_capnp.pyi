"""This is an automatically generated stub for `model.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator

class Model:
    data: bytes
    name: str
    version: str
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[ModelReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> ModelReader: ...
    @staticmethod
    def new_message() -> ModelBuilder: ...
    def to_dict(self) -> dict: ...

class ModelReader(Model):
    def as_builder(self) -> ModelBuilder: ...

class ModelBuilder(Model):
    @staticmethod
    def from_dict(dictionary: dict) -> ModelBuilder: ...
    def copy(self) -> ModelBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> ModelReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
