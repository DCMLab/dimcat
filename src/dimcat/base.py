from __future__ import annotations

import json
import logging
import os
from abc import ABC
from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from functools import cache, cached_property
from inspect import isclass
from pprint import pformat
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    overload,
)

import marshmallow as mm

# this is the only dimcat import currently allowed in this file:
from marshmallow import ValidationError, fields
from typing_extensions import Self

logger = logging.getLogger(__name__)

# ----------------------------- DEVELOPER SETTINGS -----------------------------

CONTROL_REGISTRY = False
"""Raise an error if a subclass has the same name as another subclass. Set True for production."""

# region DimcatSchema


class DtypeField(mm.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, Enum):
            return value.name
        return value


class DimcatSchema(mm.Schema):
    """
    The base class of all Schema() classes that are defined or inherited as nested classes
    for all :class:`DimcatObjects <DimcatObject>`. This class holds the logic for serializing/deserializing DiMCAT
    objects. However, nested Schema() classes should generally not inherit directly from DimcatSchema but instead
    from DimcatObject.Schema because it defines the post_load hook init_object() for deserializing Dimcat objects. The
    parent class DimcatSchema does not define it, allowing DimcatConfig to define it differently. In marshmallow,
    hooks are additive and do not replace hooks of parent schemas.

    Overall, this requires careful planning at what point in the object hierarchy the hooks are introduced in the
    corresponding nested Schema classes: Hooks of parent Schemas can be called via super() but they cannot be not
    called by omitting super(). For example, the post_dump hook validate_dump() is introduced in PipelineStep.Schema
    to automatically validate any serialized object right away (frictionless does that basically by trying if it can
    load the serialization data). For Data.Schema, however, this is not a safe default because most Data objects can
    be successfully validate only once their data has been stored to disk. Therefore, the post_dump hook is introduced
    in a second type of schema that all Data objects have, called PickleSchema.

    The arbitrary metadata of the fields currently use the keys:

    - ``expose``: Set False to mark fields that would normally not be exposed to the users in the context of a GUI.
                  Defaults to True if missing.
    - ``title``: A human-readable title for the field.
    - ``description``: A human-readable description for the field.
    """

    dtype = DtypeField(metadata={"expose": False})
    """This field specifies the class of the serialized object. Every DimcatObject comes with the corresponding class
    property that returns its name as a string (or en Enum member that can function as a string). It is inherited by
    all objects' schemas and enables their deserialization from a DimcatConfig."""

    @classmethod
    @property
    def name(cls) -> str:
        """Qualified name of the schema, meaning it includes the name of the class that it is nested in."""
        return cls.__qualname__

    class Meta:
        ordered = True

    def get_attribute(self, obj: Any, attr: str, default: Any):
        if attr == "dtype":
            # the usual marshmallow.utils.get_value() tries to access attributes by subscripting first
            # this is a problem for the DimcatConfig which behaves like a dictionary where one of the options has
            # the key 'dtype' but, when serializing, it is the property 'dtype' that is relevant
            return obj.dtype
        return super().get_attribute(obj, attr, default)

    @mm.pre_dump()
    def assert_type(self, obj, **kwargs):
        if not isinstance(obj, DimcatObject):
            raise mm.ValidationError(
                f"{self.name}: The object to be serialized needs to be a DimcatObject, not a {type(obj)!r}."
            )
        return obj

    def __repr__(self):
        return f"{self.name}(many={self.many})"

    def __getattr__(self, item):
        raise AttributeError(
            f"AttributeError: {self.name!r} object has no attribute {item!r}"
        )


# endregion DimcatSchema
# region DimcatObject


class DimcatObject(ABC):
    """All DiMCAT classes derive from DimcatObject, except for the nested Schema(DimcatSchema) class
    that they define or inherit."""

    _enum_type: ClassVar[Type[Enum]] = None
    """If a class specifies an Enum, its 'dtype' property returns the Enum member corresponding to its 'name'."""
    _registry: ClassVar[Dict[str, Type[DimcatObject]]] = {}
    """Registry of all subclasses (but not their corresponding Schema classes)."""

    @classmethod
    @property
    def dtype(cls) -> str | Enum:
        """Name of the class as enum member (if cls._enum_type is define, string otherwise)."""
        if cls._enum_type is None:
            return cls.name
        return cls._enum_type(cls.name)

    @classmethod
    @property
    def logger(cls) -> logging.Logger:
        return logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")

    @classmethod
    @property
    def name(cls) -> str:
        return str(cls.__name__)

    @classmethod
    @property
    def schema(cls):
        """Returns the (instantiated) DimcatSchema singleton object for this class."""
        return get_schema(cls.name)

    @classmethod
    def from_dict(cls, options, **kwargs) -> Self:
        """Creates a new object from a config-like dict.
        Concretely, the received ``options`` will be updated with the **kwargs and enriched with a
        'dtype' key corresponding to this object, before deserializing the dict using the
        corresponding marshmallow schema."""
        options = dict(options, **kwargs)
        if "dtype" not in options:
            cls.logger.debug(f"Added option {{'dtype': {cls.name}}}.")
            options["dtype"] = cls.name
        elif options["dtype"] != cls.name:
            cls.logger.warning(
                f"Key 'dtype' was updated from {options['dtype']} to {cls.name}."
            )
            options["dtype"] = cls.name
        try:
            return cls.schema.load(options)
        except ValidationError as e:
            msg = f"Could not instantiate {cls.name} because {cls.schema.name}, failed to validate the options:\n{e}"
            raise ValidationError(msg)

    @classmethod
    def from_config(cls, config: DimcatConfig, **kwargs) -> Self:
        """Creates a new object from a DimcatConfig.
        Concretely, the config's ``options`` will be updated with the **kwargs and the 'dtype' key
        will be replaced according to this object, before deserializing the dict using the
        corresponding marshmallow schema."""
        return cls.from_dict(config._options, **kwargs)

    @classmethod
    def from_json(cls, config: str, **kwargs) -> Self:
        json_dict = json.loads(config)
        return cls.from_dict(json_dict, **kwargs)

    @classmethod
    def from_json_file(cls, filepath: str) -> Self:
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.from_json(json_data)

    class PickleSchema:
        def __init__(self):
            raise NotImplementedError(
                "This object does not support automatic pickling to a basepath (yet)."
            )

    class Schema(DimcatSchema):
        @mm.post_load()
        def init_object(self, data, **kwargs) -> DimcatObject:
            """Once the data has been loaded, create the corresponding object."""
            obj_name = data.pop("dtype")
            Constructor = get_class(obj_name)
            return Constructor(**data)

    def __init__(self):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        if CONTROL_REGISTRY and cls.name in cls._registry:
            raise RuntimeError(
                f"A class named {cls.name!r} had already been registered. Choose a different name."
            )
        cls._registry[cls.name] = cls

    def __eq__(self, other):
        if not isinstance(other, DimcatObject):
            return False
        if other.name != self.name:
            return False
        return other.to_dict() == self.to_dict()

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self):
        return self.info(return_str=True)

    def __str__(self):
        return f"{__name__}.{self.name}"

    def filename_factory(self):
        return self.name

    @overload
    def info(self, return_str: Literal[False]) -> None:
        ...

    @overload
    def info(self, return_str: Literal[True]) -> str:
        ...

    def info(self, return_str: bool = False) -> Optional[str]:
        """Returns a summary of the dataset."""
        summary = self.summary_dict()
        title = self.name
        title += f"\n{'=' * len(title)}\n"
        summary_str = f"{title}{pformat(summary, sort_dicts=False, width=120)}"
        if return_str:
            return summary_str
        print(summary_str)

    def summary_dict(self) -> dict:
        """Returns a summary of the object."""
        return self.to_dict()

    def to_dict(self) -> dict:
        return self.schema.dump(self)

    def to_config(self) -> DimcatConfig:
        return DimcatConfig(self.to_dict())

    def to_options(self):
        D = self.to_dict()
        del D["dtype"]
        return D

    def to_json(self) -> str:
        return self.schema.dumps(self)

    def to_json_file(self, filepath: str, indent: int = 2, **kwargs):
        """Serialize object to JSON file.

        Args:
            filepath: Path to the text file to (over)write.
            indent: Prettify the JSON layout. Default indentation: 2 spaces
            **kwargs: Keyword arguments passed to :meth:`json.dumps`.
        """
        as_dict = self.to_dict()
        as_dict.update(**kwargs)
        if is_default_descriptor_path(filepath):
            frictionless = (
                ""
                if not hasattr(self, "store_descriptor")
                else f" Use {self.name}.store_descriptor() "
                f"to store a frictionless descriptor."
            )
            raise ValueError(
                f"The JSON path {filepath!r} corresponds to the name of frictionless descriptor and "
                f"mustn't be used for a 'normal' DiMCAT serialization.{frictionless}"
            )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(as_dict, f, indent=indent, **kwargs)
        self.logger.info(f"{self.name} has been serialized to {filepath!r}")


class DimcatObjectField(fields.Field):
    """Used for (de)serializing attributes resolving to DimcatObjects."""

    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, DimcatConfig):
            return dict(value)
        return value.to_dict()

    def _deserialize(self, value, attr, data, **kwargs):
        return deserialize_dict(value)


class FriendlyEnum(str, Enum):
    """Members of this Enum can be created from and compared to strings in a case-insensitive
    manner."""

    @classmethod
    def _missing_(cls, value) -> Self:
        value_lower = value.lower()
        lc_values = {member.value.lower(): member for member in cls}
        if value_lower in lc_values:
            return lc_values[value_lower]
        for lc_value, member in lc_values.items():
            if lc_value.startswith(value_lower):
                return member
        raise ValueError(f"ValueError: {value!r} is not a valid {cls.__name__}.")

    def __eq__(self, other) -> bool:
        if self.value == other:
            return True
        if isinstance(other, str):
            return other.lower() == self.value.lower()
        return False

    def __hash__(self):
        return hash(self.value)


class ObjectEnum(FriendlyEnum):
    @cache
    def get_class(self) -> Type[DimcatObject]:
        return get_class(self.name)


# endregion DimcatObject
# region DimcatConfig


class DimcatConfig(MutableMapping, DimcatObject):
    """Behaves like a dictionary but accepts only keys and values that are valid under the Schema of the DimcatObject
    specified under the key 'dtype'. Every DimcatConfig needs to have a 'dtype' key that is the name of a DimcatObject
    and can specify zero or more additional key-value pairs that can be used to instantiate the described object.

    When dealing with a DimcatConfig you need to be aware that the 'dtype' and 'options' can been different things
    when used as keys as opposed to attributes (``DC`` represents a DimcatConfig):

    - ``DC['dtype']`` is the name of the described DimcatObject (equivalent to ``DC.options_dtype``)
    - ``DC.dtype`` returns the class name "DimcatConfig", according to all DimcatObjects' default behaviour
    - ``DC.options`` (equivalent to ``dict(DC)`` returns the key-value pairs wrapped by this config,
      which includes at least the 'dtype' key
    - ``DC['options']`` is the value of the 'options' option, which exists only if it is part of the described
      object's schema, for example if the described object is a :class:`DimcatConfig` itself.

    Examples:

        >>> from dimcat.base import DimcatConfig, DimcatObject
        >>> DC = DimcatConfig(dtype="DimcatObject")
        >>> DC.dtype
        'DimcatConfig'
        >>> DC['dtype']
        'DimcatObject'
        >>> DC.options
        {'dtype': 'DimcatObject'}
        >>> DC['options']
        KeyError: 'options'
        >>> config_config = DC.to_config()
        >>> config_config.options
        {'dtype': 'DimcatConfig', 'options': {'dtype': 'DimcatObject'}}
        >>> config_config['options']
        {'dtype': 'DimcatObject'}

    """

    class Schema(DimcatSchema):
        options = mm.fields.Dict()

        @mm.pre_load()
        def serialize_if_necessary(self, data, **kwargs):
            if isinstance(data, DimcatObject):
                return dict(options=data.options)
            return data

        @mm.post_load()
        def init_object(self, data, **kwargs) -> DimcatConfig:
            """Once the data has been loaded, create the corresponding object."""
            Constructor = get_class("DimcatConfig")
            return Constructor(**data)

        @mm.post_dump()
        def validate_dump(self, data, **kwargs):
            """Make sure to never return invalid serialization data."""
            if "dtype" not in data:
                raise mm.ValidationError(
                    "The object to be serialized doesn't have a 'dtype' field. May it's not a "
                    "DimcatObject?"
                )
            if data["dtype"] != "DimcatConfig":
                raise mm.ValidationError(
                    f"The object was serialized as a {data['dtype']} rather than a DimcatConfig: {data}"
                )
            options_dtype = data["options"]["dtype"]
            dtype_schema = get_schema(options_dtype)
            report = dtype_schema.validate(data["options"])
            if report:
                raise mm.ValidationError(
                    f"Dump of DimcatConfig(dtype={options_dtype}) created with a {self.name} could not be "
                    f"validated by {dtype_schema.name} :\n{report}"
                )
            return dict(data)

    def __init__(
        self, options: Dict | DimcatConfig = (), dtype: Optional[str] = None, **kwargs
    ):
        if isinstance(options, DimcatConfig):
            options = options.options
        elif isinstance(options, str) and dtype is None:
            options = dict(dtype=options)
        options = dict(options, **kwargs)
        if dtype is None:
            if "dtype" not in options:
                raise ValidationError(
                    "DimcatConfig requires a 'dtype' key that needs to be the name of a DimcatObject."
                )
            else:
                dtype = options["dtype"]
        else:
            if dtype == "DimcatConfig":
                if "options" not in options:
                    options = dict(dtype="DimcatConfig", options=options)
            elif "dtype" not in options:
                options["dtype"] = dtype
        if dtype is None:
            raise ValidationError(
                "'dtype' key cannot be None, it needs to be the name of a DimcatObject."
            )
        if not is_name_of_dimcat_class(dtype):
            raise ValidationError(
                f"'dtype' key needs to be the name of a DimcatObject, not {dtype!r}. Registry:\n"
                f"{DimcatObject._registry}"
            )
        self._options: dict = options
        """The options dictionary wrapped and controlled by this DimcatConfig. Whenever a new value is set, it is
        validated against the Schema of the DimcatObject specified under the key 'dtype'."""
        if (
            isinstance(dtype, Enum)
            or isinstance(dtype, DimcatObject)
            or (isclass(dtype) and issubclass(dtype, DimcatObject))
        ):
            dtype_str = dtype.name
            self._options["dtype"] = dtype_str
        elif isinstance(dtype, str):
            pass
        else:
            raise ValueError(
                f"{dtype!r} is not the name of a DimcatObject, needed to instantiate a Config."
            )
        report = self.validate(partial=True)
        if report:
            raise ValidationError(
                f"{self.options_schema}: Cannot instantiate DimcatConfig with dtype={dtype!r} and invalid options:"
                f"\n{report}"
            )

    def __delitem__(self, key):
        if key == "dtype":
            raise ValueError("Cannot remove key 'dtype' from DimcatConfig.")
        del self._options[key]

    def __eq__(self, other: DimcatObject | MutableMapping) -> bool:
        """The comparison with another DimcatConfig or dict-like returns True if both describe the same object or if
        one describes the other. That is,
        - both describe the same object, i.e. key 'dtype' is the same and any other options are identical, or
        - this DimcatConfig describes the other object, or
        - the other object describes this DimcatConfig.
        """
        if isinstance(other, DimcatConfig):
            other = other.options
        elif isinstance(other, DimcatObject):
            if other.name != self.options_dtype:
                return False
            return other.to_dict() == self._options

        if not isinstance(other, MutableMapping):
            raise TypeError(
                f"{self.name} can only be compared against dict-like, not {type(other)}."
            )
        if "dtype" not in other:
            return False
        self_describes_config, other_describes_config = (
            self["dtype"] == "DimcatConfig",
            other["dtype"] == "DimcatConfig",
        )
        if self_describes_config == other_describes_config:
            return self.options == other
        # exactly one of the two described a DimcatConfig, hance we check if one is a serialized version of the other
        a, b = (self, other) if self_describes_config else (other, self)
        return self["options"] == other

    def __getitem__(self, key):
        return self._options[key]

    def __iter__(self):
        return iter(self._options)

    def __len__(self):
        return len(self._options)

    def __setitem__(self, key, value):
        if key == "dtype" and value != self._options["dtype"]:
            tmp_schema = get_schema(value)
            tmp_dict = dict(self._options, dtype=value)
            report = tmp_schema.validate(tmp_dict)
            if report:
                msg = (
                    f"Cannot change the value for 'dtype' because its {tmp_schema.name} does not "
                    f"validate the options:\n{report}"
                )
                raise ValidationError(msg)
        else:
            dict_to_validate = {key: value}
            report = self.options_schema.validate(dict_to_validate, partial=True)
            if report:
                msg = f"{self.options_schema.name}: Cannot set {key!r} to {value!r}:\n{report}"
                raise ValidationError(msg)
        self._options[key] = value

    @cached_property
    def options_class(self):
        """The class of the described DimcatObject."""
        return get_class(self.options_dtype)

    @property
    def options_dtype(self) -> str:
        """The dtype (i.e. class name) of the described DimcatObject."""
        return self._options["dtype"]

    @property
    def options_schema(self):
        """Returns the (instantiated) Dimcat singleton object for the class this Config describes."""
        return get_schema(self.options_dtype)

    @classmethod
    def from_dict(cls, options, **kwargs) -> Self:
        return cls(options, **kwargs)

    @classmethod
    def from_object(cls, obj: DimcatObject):
        options = obj.to_dict()
        return cls(options)

    @property
    def options(self):
        """Returns the options dictionary wrapped and controlled by this DimcatConfig. Whenever a new value is set,
        it is validated against the Schema of the DimcatObject specified under the key 'dtype'. Note that this property
        returns a copy of the dictionary including the 'dtype' key and modifying it will not affect the DimcatConfig.
        Also note that the returned value is different from DimcatConfig["options"]"""
        return dict(self._options)

    def create(self) -> DimcatObject:
        return self.options_schema.load(self._options)

    def matches(self, config: DimcatConfig) -> bool:
        """Returns True if both configs have the same :attr:`options_dtype` and the overlapping options are equal."""
        if not isinstance(config, DimcatConfig):
            raise TypeError(
                f"Can only compare against DimcatConfig, not {type(config)}."
            )
        if self.options_dtype != config.options_dtype:
            return False
        overlapping_keys = set(self.options.keys()) & set(config.options.keys())
        for key in overlapping_keys:
            if self[key] != config[key]:
                return False
        return True

    def summary_dict(self) -> dict:
        return self._options

    def validate(self, partial=False) -> Dict[str, List[str]]:
        """Validates the current status of the config in terms of ability to create an object. Empty dict == valid."""
        return self.options_schema.validate(self._options, many=False, partial=partial)


# endregion DimcatConfig
# region Data and PipelineStep


# endregion Data and PipelineStep
# region querying DimcatObjects by name
@cache
def get_class(name) -> Type[DimcatObject]:
    if isinstance(name, Enum):
        name = name.name
    if name == "DimcatObject":
        # this is the only object that's not in the registry
        return DimcatObject
    try:
        return DimcatObject._registry[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not among the registered DimcatObjects:\n{DimcatObject._registry.keys()}"
        )


@cache
def is_name_of_dimcat_class(name) -> bool:
    """"""
    try:
        get_class(name)
        return True
    except KeyError:
        return False


def is_instance_of(obj, class_or_tuple: Type | Tuple[Type | str, ...]):
    """Returns True if the given object is an instance of the given class or one of the given classes."""
    if not isinstance(class_or_tuple, tuple):
        class_or_tuple = (class_or_tuple,)
    classes = tuple(get_class(c) if isinstance(c, str) else c for c in class_or_tuple)
    return isinstance(obj, classes)


@cache
def is_subclass_of(name: str, parent: str | Type[DimcatObject]) -> bool:
    """Returns True if the DimcatObject with the given name is a subclass of the given parent."""
    cls = get_class(name)
    if isinstance(parent, str):
        parent = get_class(parent)
    return issubclass(cls, parent)


@cache
def get_pickle_schema(name, init=True):
    """Caches the intialized schema for each class. Pass init=False to retrieve the schema constructor."""
    dc_class = get_class(name)
    dc_schema = dc_class.PickleSchema
    if init:
        return dc_schema()
    return dc_schema


@cache
def get_schema(name, init=True):
    """Caches the intialized schema for each class. Pass init=False to retrieve the schema constructor."""
    dc_class = get_class(name)
    dc_schema = dc_class.Schema
    if init:
        return dc_schema()
    return dc_schema


def deserialize_config(config) -> DimcatObject:
    """Deserialize a config object into a DimcatObject."""
    return config.create()


def deserialize_dict(obj_data) -> DimcatObject:
    """Deserialize a dict into a DimcatObject."""
    config = DimcatConfig(obj_data)
    return deserialize_config(config)


def deserialize_json_str(json_data) -> DimcatObject:
    """Deserialize a JSON string into a DimcatObject."""
    obj_data = json.loads(json_data)
    return deserialize_dict(obj_data)


def deserialize_json_file(json_file) -> DimcatObject:
    """Deserialize a JSON file into a DimcatObject."""
    with open(json_file, "r") as f:
        json_data = f.read()
    return deserialize_json_str(json_data)


# endregion querying DimcatObjects by name
# region DimcatSettings


@dataclass
class DimcatSettings(DimcatObject):
    """This is a dataclass that stores the default settings for the dimcat library.
    The current settings can be loaded anywhere in the code by calling :func:`get_setting`.
    Defining a new setting means adding it in three places:

    1. as a class attribute with type annotation (=dataclass field)  and default (factory)
    2. as a marshmallow field in the Schema with the same name and corresponding type, which
       gives access to marshmallow's full serialization and validation functionality. By default,
       we add ``required=True`` to all settings.
    3. to the file ``settings.ini``, using Python's config file syntax
    """

    auto_make_dirs: bool = True
    context_columns: List[str] = dataclass_field(
        default_factory=lambda: [
            "mc",
            "mn",
            "quarterbeats",
            "quarterbeats_all_endings",
            "duration_qb",
            "duration",
            "mc_onset",
            "mn_onset",
            "timesig",
            "staff",
            "voice",
            "volta",
        ]
    )
    """the columns that are considered essential for locating elements horizontally and vertically and which are
    therefore always copied, if present, and moved to the left of the new dataframe in the order given here"""
    default_basepath: str = "~/dimcat_data"
    """where to serialize data if no other basepath is specified"""
    default_figure_path: str = "~/dimcat_data"
    """where to store figures if no other path was specified"""
    default_figure_format: str = ".png"
    """default format for all figures stored by DiMCAT."""
    default_figure_width: int = 2880
    """default width in pixels for figures stored by DiMCAT"""
    default_figure_height: int = 1620
    """default height in pixels for figures stored by DiMCAT"""
    default_resource_name: str = "unnamed"
    never_store_unvalidated_data: bool = True
    """setting this to False allows for skipping mandatory validations; set to True for production"""
    recognized_piece_columns: List[str] = dataclass_field(
        default_factory=lambda: ["piece", "pieces", "fname", "fnames"]
    )
    """column names that are recognized as piece identifiers and automatically renamed to 'piece' when needed"""
    package_descriptor_endings: List[str] = dataclass_field(
        default_factory=lambda: ["package.json", "package.yaml"]
    )
    resource_descriptor_endings: List[str] = dataclass_field(
        default_factory=lambda: ["resource.json", "resource.yaml"]
    )
    """file endings that are recognized as frictionless resource descriptors"""

    class Schema(DimcatObject.Schema):
        auto_make_dirs = mm.fields.Boolean(
            required=True,
            metadata={
                "description": "whether to automatically create directories, e.g. when indicating basepaths"
            },
        )
        context_columns = mm.fields.List(
            mm.fields.String(),
            required=True,
            metadata={
                "description": "the columns that are considered essential for locating elements horizontally and "
                "vertically and which are therefore always copied, if present, and moved to the left "
                "of the new dataframe in the given order"
            },
        )
        default_basepath = mm.fields.String(
            required=True,
            metadata={
                "description": "where to serialize data if no other basepath is specified"
            },
        )
        default_figure_path = mm.fields.String(
            required=True,
            metadata={
                "description": "where to store figures if no other path was specified"
            },
        )
        default_figure_format = mm.fields.String(
            required=True,
            metadata={"description": "default format for all figures stored by DiMCAT"},
        )
        default_figure_width = mm.fields.Integer(
            required=True,
            metadata={
                "description": "default width in pixels for figures stored by DiMCAT"
            },
        )
        default_figure_height = mm.fields.Integer(
            required=True,
            metadata={
                "description": "default height in pixels for figures stored by DiMCAT"
            },
        )
        default_resource_name = mm.fields.String(required=True)
        never_store_unvalidated_data = mm.fields.Boolean(
            required=True,
            metadata={
                "description": "setting this to False allows for "
                "skipping mandatory validations; set to True for production"
            },
        )
        recognized_piece_columns = mm.fields.List(
            mm.fields.String(),
            required=True,
            metadata={
                "description": "column names that are recognized as piece "
                "identifiers and automatically renamed to 'piece' when needed"
            },
        )
        package_descriptor_endings = mm.fields.List(
            mm.fields.String(),
            required=True,
            metadata={
                "description": "file endings that are recognized as frictionless package descriptors"
            },
        )
        resource_descriptor_endings = mm.fields.List(
            mm.fields.String(),
            required=True,
            metadata={
                "description": "file endings that are recognized as frictionless resource descriptors"
            },
        )


def parse_config_file(config_filepath: str) -> ConfigParser:
    """Parse a config file and return a ConfigParser object."""
    if not os.path.isfile(config_filepath):
        raise FileNotFoundError(f"Config file '{config_filepath}' not found.")
    config = ConfigParser()
    config.read(config_filepath)
    return config


def make_default_settings() -> DimcatConfig:
    """Make a DimcatConfig object representing DimcatSettings with default values."""
    return DimcatSettings().to_config()


def make_settings_from_config_parser(config: ConfigParser) -> DimcatConfig:
    """Make a DimcatSettings object from a ConfigParser object."""
    settings = make_default_settings()
    # recognized_settings = [f.name for f in dataclass_fields(settings)]
    all_section_names: list[str] = config.sections()
    all_section_names.append("DEFAULT")
    setting_fields = {
        name: f for name, f in DimcatSettings.schema.declared_fields.items()
    }
    for section_name in all_section_names:
        for key, value in config.items(section_name):
            split_value = [
                word for line in value.split("\n") for word in line.split(",")
            ]
            while "" in split_value:
                split_value.remove("")
            if len(split_value) > 1:
                settings[key] = split_value
            else:
                try:
                    setting_field = setting_fields[key]
                except KeyError:
                    logger.warning(
                        f"Ignoring unrecognized setting '{key}'. In order to add it to the library, "
                        f"it needs to be added to the DimcatSettings dataclass and to its Schema."
                    )
                    continue
                if isinstance(setting_field, mm.fields.Boolean):
                    value = value in setting_field.truthy
                settings[key] = value
    return settings


def make_settings_from_config_file(
    config_filepath: str,
    fallback_to_default: bool = True,
) -> DimcatConfig:
    """Make a DimcatSettings object from a config file."""
    try:
        config = parse_config_file(config_filepath)
    except FileNotFoundError:
        if not fallback_to_default:
            raise
        logger.error(
            f"Config file '{config_filepath}' not found. Falling back to default settings."
        )
        return make_default_settings()
    try:
        return make_settings_from_config_parser(config)
    except Exception as e:
        if not fallback_to_default:
            raise
        logger.error(
            f"Error while parsing config file '{config_filepath}': {e}. Falling back to default settings."
        )
        return make_default_settings()


def load_settings(
    config_filepath: Optional[str] = None,
    raise_exception: bool = False,
) -> DimcatConfig:
    """Get the DimcatSettings object."""
    fallback_to_default = not raise_exception
    if config_filepath is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = os.path.join(dir_path, "settings.ini")
        if os.path.isfile(config_filepath):
            settings = make_settings_from_config_file(
                config_filepath, fallback_to_default=fallback_to_default
            )
            logger.info(f"Loaded default config file at '{config_filepath}'.")
            return settings
        elif fallback_to_default:
            logger.warning(
                f"No config file path was provided and the default config file was not found: "
                f"{config_filepath}. Falling back to default."
            )
            return make_default_settings()
        else:
            raise FileNotFoundError(
                f"No config file path was provided and the default config file was not found: "
                f"{config_filepath}. Falling back to default."
            )
    settings = make_settings_from_config_file(
        config_filepath, fallback_to_default=fallback_to_default
    )
    logger.info(f"Loaded config file at '{config_filepath}'.")
    return settings


SETTINGS: DimcatConfig = load_settings()


def get_setting(key: str) -> Any:
    return SETTINGS[key]


def change_setting(key: str, value: Any) -> None:
    SETTINGS[key] = value
    logger.info(f"Changed setting {key!r} to {value!r}.")


def reset_settings(config_filepath: Optional[str] = None) -> None:
    """Reset the DiMCAT settings to the default or to those found in the settings.ini file at the
    given path."""
    global SETTINGS
    SETTINGS = load_settings(config_filepath)


# endregion DimcatSettings
# region helper function


def is_default_descriptor_path(filepath: str) -> bool:
    rde = get_setting("resource_descriptor_endings")
    pde = get_setting("package_descriptor_endings")
    default_descriptor_endings = rde + pde
    for ending in default_descriptor_endings:
        if filepath.endswith(ending):
            return True
    return False


# endregion helper function
