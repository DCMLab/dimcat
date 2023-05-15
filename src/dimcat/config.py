from collections.abc import MutableMapping
from typing import Dict, List

import marshmallow as mm
from dimcat.base import DimcatObject
from dimcat.utils import get_class, get_schema


class DimcatSchema(mm.Schema):
    """
    The base class of the config system of Dimcat, with which to define custom configs. This class holds the logic for
    serializing/deserializing config objects.
    """

    dtype = mm.fields.String()

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__qualname__

    @mm.post_load()
    def init_object(self, data, **kwargs):
        obj_name = data.pop("dtype")
        Constructor = get_class(obj_name)
        return Constructor(**data)

    @mm.post_dump()
    def validate_dump(self, data, **kwargs) -> None:
        if "dtype" not in data:
            raise mm.ValidationError(
                "The object to be serialized doesn't have a 'dtype' field. May it's not a "
                "DimcatObject?"
            )
        dtype_schema = get_schema(data["dtype"])
        report = dtype_schema.validate(data)
        if report:
            raise mm.ValidationError(
                f"Dump of {data['dtype']} created with a {self.name} could not be validated by "
                f"{dtype_schema.name} :\n{report}"
            )
        return data

    def __repr__(self):
        return f"{self.name}(many={self.many})"


class DimcatConfig(MutableMapping, DimcatObject):
    def __init__(self, options=(), **kwargs):
        self._config = dict(options, **kwargs)
        if "dtype" not in self._config:
            raise ValueError(
                "DimcatConfig requires a 'dtype' key that needs to be the name of a DimcatObject."
            )
        dtype = self._config["dtype"]
        if isinstance(dtype, str):
            dtype_str = dtype
        elif isinstance(dtype, DimcatObject) or issubclass(dtype, DimcatObject):
            dtype_str = dtype.name
            self._config["dtype"] = dtype_str
        else:
            raise ValueError(
                f"{dtype!r} is not the name of a DimcatObject, needed to instantiate a Config."
            )
        self.schema: DimcatSchema = get_schema(dtype_str)
        self._config = dict(dtype=dtype_str)
        self.update(options)

    @classmethod
    def from_object(cls, obj: DimcatObject):
        dump = obj.schema.dump(obj)
        return cls(dump)

    def create(self) -> DimcatObject:
        return self.schema.load(self._config)

    def validate(self) -> Dict[str, List[str]]:
        """Validates the current status of the config in terms of ability to create an object. Empty dict == valid."""
        return self.schema.validate(self._config, many=False, partial=False)

    def __getitem__(self, key):
        return self._config[key]

    def __delitem__(self, key):
        del self._config[key]

    def __setitem__(self, key, value):
        dict_to_validate = {key: value}
        report = self.schema.validate(dict_to_validate, partial=True)
        if report:
            raise ValueError(
                f"{self.schema.name}: Cannot set {key!r} to {value!r}:\n{report}"
            )
        self._config[key] = value

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __repr__(self):
        return f"{self.name}({self._config})"
