import json

import pydantic
import pytest

from smartsim._core.schemas.utils import SchemaSerializer


class Person(pydantic.BaseModel):
    name: str
    age: int


class Book(pydantic.BaseModel):
    title: str
    num_pages: int


def test_schema_registrartion():
    serializer = SchemaSerializer("test_type")
    assert serializer._map == {}

    serializer.register("person")(Person)
    assert serializer._map == {"person": Person}

    serializer.register("book")(Book)
    assert serializer._map == {"person": Person, "book": Book}


def test_serialize_schema():
    serializer = SchemaSerializer("test_type")
    serializer.register("person")(Person)
    serializer.register("book")(Book)
    assert json.loads(serializer.serialize_to_json(Person(name="Bob", age=36))) == {
        "__test_type__": "person",
        "name": "Bob",
        "age": 36,
    }
    assert json.loads(
        serializer.serialize_to_json(
            Book(title="The Greatest Story of All Time", num_pages=10_000)
        )
    ) == {
        "__test_type__": "book",
        "title": "The Greatest Story of All Time",
        "num_pages": 10_000,
    }


def test_serializer_errors_if_types_overloaded():
    serializer = SchemaSerializer("test_type")
    serializer.register("schema")(Person)

    with pytest.raises(KeyError):
        serializer.register("schema")(Book)


def test_serializer_errors_on_unknown_schema():
    serializer = SchemaSerializer("test_type")
    serializer.register("person")(Person)

    with pytest.raises(TypeError):
        serializer.serialize_to_json(
            Book(title="The Shortest Story of All Time", num_pages=1)
        )


def test_deserialize_json():
    serializer = SchemaSerializer("test_type")
    serializer.register("person")(Person)
    serializer.register("book")(Book)
    assert serializer.deserialize_from_json(
        json.dumps({"__test_type__": "person", "name": "Bob", "age": 36})
    ) == Person(name="Bob", age=36)
    assert serializer.deserialize_from_json(
        json.dumps(
            {
                "__test_type__": "book",
                "title": "The Most Average Story of All Time",
                "num_pages": 500,
            }
        )
    ) == Book(title="The Most Average Story of All Time", num_pages=500)


def test_deserialize_error_if_type_key_not_recognized():
    serializer = SchemaSerializer("test_type")
    serializer.register("person")(Person)

    with pytest.raises(ValueError):
        serializer.deserialize_from_json(
            json.dumps(
                {"__test_type__": "alien", "name": "Bob the Alien", "age": 5_000}
            )
        )


def test_deserialize_error_if_type_key_is_missing():
    serializer = SchemaSerializer("test_type")
    serializer.register("person")(Person)

    with pytest.raises(ValueError):
        serializer.deserialize_from_json(json.dumps({"name": "Bob", "age": 36}))
