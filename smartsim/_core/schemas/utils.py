import json
import typing as t

import pydantic

_KeyT = t.TypeVar("_KeyT")
_SchemaT = t.TypeVar("_SchemaT", bound=pydantic.BaseModel)


class SchemaSerializer(t.Generic[_KeyT, _SchemaT]):
    def __init__(
        self,
        type_name: str,
        init_map: t.Optional[t.Mapping[_KeyT, t.Type[_SchemaT]]] = None,
    ):
        self._map = dict(init_map) if init_map else {}
        self._type_name_key = f"__{type_name}__"

    def register(self, key: _KeyT) -> t.Callable[[t.Type[_SchemaT]], t.Type[_SchemaT]]:
        if key in self._map:
            raise KeyError(f"Key `{key}` has already been registered for this parser")

        def _register(cls: t.Type[_SchemaT]) -> t.Type[_SchemaT]:
            self._map[key] = cls
            return cls

        return _register

    def schema_to_dict(self, schema: _SchemaT) -> t.Dict[str, t.Any]:
        reverse_map = dict((v, k) for k, v in self._map.items())
        try:
            val = reverse_map[type(schema)]
        except KeyError:
            raise TypeError(f"Unregistered schema type: {type(schema)}") from None
        # TODO: This method is deprectated in pydantic >= 2
        dict_ = schema.dict()
        dict_[self._type_name_key] = val
        return dict_

    def serialize_to_json(self, schema: _SchemaT) -> str:
        return json.dumps(self.schema_to_dict(schema))

    def mapping_to_schema(self, obj: t.Mapping[t.Any, t.Any]) -> _SchemaT:
        try:
            type_ = obj[self._type_name_key]
        except KeyError:
            raise ValueError(f"Could not parse object: {obj}") from None
        try:
            cls = self._map[type_]
        except KeyError:
            raise ValueError(f"No type of value `{type_}` is registered") from None
        return cls.parse_obj(obj)

    def deserialize_from_json(self, obj: str) -> _SchemaT:
        return self.mapping_to_schema(json.loads(obj))
