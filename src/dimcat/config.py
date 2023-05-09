from typing import TYPE_CHECKING

import marshmallow as mm

# To avoid circular import.
if TYPE_CHECKING:
    from dimcat.base import DimcatObject

_CONFIGURED_REGISTRY = {}


class DimCatConfig(mm.Schema):
    """
    The base class of the config system of Dimcat, with which to define custom configs. This class holds the logic for
    serializing/deserializing config objects.
    """

    # Contains the configured class. This class is returned when loading a config object.
    # This should be overridden by every subclass.
    configured_object: "DimcatObject" = None
    # This is used internally by marshmallow to deserialize the configured object.
    # Note that this field added "manually" by add_class_builder_to_json, for practical reasons.
    _configured_type = mm.fields.Function(deserialize=lambda v : _CONFIGURED_REGISTRY[v])

    def __new__(cls, *args, **kwargs):
        # Register a new class builder to the _CLASS_BUILDERS registry. Is used to serialize/deserialize dim cats objects
        # configured.
        if cls.configured_object is None:
            raise ValueError(f"{cls} object does not have configured_object attribute !")
        _CONFIGURED_REGISTRY[cls.configured_object.__qualname__] = cls.configured_object
        return object.__new__(cls)

    @mm.post_dump
    def add_class_builder_to_json(self, data, **kwargs):
        # Add the field containing the configured object class, that will be used a class builder upon deserializing.
        data["_configured_type"] = self.configured_object.__qualname__
        return data

    @classmethod
    def from_json(cls, json) -> dict:
        # Instantiating cls() everytime is is quite ugly. There is likely a better solution
        return cls().loads(json)