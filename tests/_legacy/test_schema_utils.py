# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections
import json

import pydantic
import pytest

from smartsim._core.schemas.utils import (
    _DEFAULT_MSG_DELIM,
    SchemaRegistry,
    SocketSchemaTranslator,
    _Message,
)

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


class Person(pydantic.BaseModel):
    name: str
    age: int


class Dog(pydantic.BaseModel):
    name: str
    age: int


class Book(pydantic.BaseModel):
    title: str
    num_pages: int


def test_equivalent_messages_are_equivalent():
    book = Book(title="A Story", num_pages=250)
    msg_1 = _Message(book, "header")
    msg_2 = _Message(book, "header")

    assert msg_1 is not msg_2
    assert msg_1 == msg_2
    assert str(msg_1) == str(msg_2)
    assert msg_1 == _Message.from_str(str(msg_1), Book)


def test_schema_registrartion():
    registry = SchemaRegistry()
    assert registry._map == {}

    registry.register("person")(Person)
    assert registry._map == {"person": Person}

    registry.register("book")(Book)
    assert registry._map == {"person": Person, "book": Book}


def test_cannot_register_a_schema_under_an_empty_str():
    registry = SchemaRegistry()
    with pytest.raises(KeyError, match="Key cannot be the empty string"):
        registry.register("")


def test_schema_to_string():
    registry = SchemaRegistry()
    registry.register("person")(Person)
    registry.register("book")(Book)
    person = Person(name="Bob", age=36)
    book = Book(title="The Greatest Story of All Time", num_pages=10_000)
    assert registry.to_string(person) == str(_Message(person, "person"))
    assert registry.to_string(book) == str(_Message(book, "book"))


def test_schemas_with_same_shape_are_mapped_correctly():
    registry = SchemaRegistry()
    registry.register("person")(Person)
    registry.register("dog")(Dog)

    person = Person(name="Mark", age=34)
    dog = Dog(name="Fido", age=5)

    parsed_person = registry.from_string(registry.to_string(person))
    parsed_dog = registry.from_string(registry.to_string(dog))

    assert isinstance(parsed_person, Person)
    assert isinstance(parsed_dog, Dog)

    assert parsed_person == person
    assert parsed_dog == dog


def test_registry_errors_if_types_overloaded():
    registry = SchemaRegistry()
    registry.register("schema")(Person)

    with pytest.raises(KeyError):
        registry.register("schema")(Book)


def test_registry_errors_if_msg_type_registered_with_delim_present():
    registry = SchemaRegistry()
    with pytest.raises(ValueError, match="cannot contain delimiter"):
        registry.register(f"some_key_with_the_{_DEFAULT_MSG_DELIM}_as_a_substring")


def test_registry_errors_on_unknown_schema():
    registry = SchemaRegistry()
    registry.register("person")(Person)

    with pytest.raises(TypeError):
        registry.to_string(Book(title="The Shortest Story of All Time", num_pages=1))


def test_registry_correctly_maps_to_expected_type():
    registry = SchemaRegistry()
    registry.register("person")(Person)
    registry.register("book")(Book)
    person = Person(name="Bob", age=36)
    book = Book(title="The Most Average Story of All Time", num_pages=500)
    assert registry.from_string(str(_Message(person, "person"))) == person
    assert registry.from_string(str(_Message(book, "book"))) == book


def test_registery_errors_if_type_key_not_recognized():
    registry = SchemaRegistry()
    registry.register("person")(Person)

    with pytest.raises(ValueError, match="^No type of value .* registered$"):
        registry.from_string(str(_Message(Person(name="Grunk", age=5_000), "alien")))


def test_registry_errors_if_type_key_is_missing():
    registry = SchemaRegistry()
    registry.register("person")(Person)

    with pytest.raises(ValueError, match="Failed to determine schema type"):
        registry.from_string("This string does not contain a delimiter")


class MockSocket:
    def __init__(self, send_queue, recv_queue):
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    def send_string(self, str_, *_args, **_kwargs):
        assert isinstance(str_, str)
        self.send_queue.append(str_)

    def recv_string(self, *_args, **_kwargs):
        str_ = self.recv_queue.popleft()
        assert isinstance(str_, str)
        return str_


class Request(pydantic.BaseModel): ...


class Response(pydantic.BaseModel): ...


def test_socket_schema_translator_uses_schema_registries():
    server_to_client = collections.deque()
    client_to_server = collections.deque()

    server_socket = MockSocket(server_to_client, client_to_server)
    client_socket = MockSocket(client_to_server, server_to_client)

    req_reg = SchemaRegistry()
    res_reg = SchemaRegistry()

    req_reg.register("message")(Request)
    res_reg.register("message")(Response)

    server = SocketSchemaTranslator(server_socket, res_reg, req_reg)
    client = SocketSchemaTranslator(client_socket, req_reg, res_reg)

    # Check sockets are able to communicate seamlessly with schemas only
    client.send(Request())
    assert len(client_to_server) == 1
    req = server.recv()
    assert len(client_to_server) == 0
    assert isinstance(req, Request)

    server.send(Response())
    assert len(server_to_client) == 1
    res = client.recv()
    assert len(server_to_client) == 0
    assert isinstance(res, Response)

    # Ensure users cannot send unexpected schemas
    with pytest.raises(TypeError, match="Unregistered schema"):
        client.send(Response())
    with pytest.raises(TypeError, match="Unregistered schema"):
        server.send(Request())
