from __future__ import annotations

import json
import logging
from abc import ABC
from enum import Enum
from functools import cache
from inspect import isclass
from pprint import pformat
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import marshmallow as mm
from marshmallow import ValidationError
from typing_extensions import Self

logger = logging.getLogger(__name__)

# ----------------------------- GLOBAL SETTINGS -----------------------------

NEVER_STORE_UNVALIDATED_DATA = (
    False  # allows for skipping mandatory validations; set to True for production
)
CONTROL_REGISTRY = (
    False  # raise an error if a subclass has the same name as another subclass
)


class DtypeField(mm.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, Enum):
            return value.name
        return value


class DimcatSchema(mm.Schema):
    """
    The base class of all Schema() classes that are defined or inherited as nested classes
    for all :class:`DimcatObjects <DimcatObject>`. This class holds the logic for serializing/deserializing DiMCAT
    objects.
    """

    dtype = DtypeField()
    """This field specifies the class of the serialized object. Every DimcatObject comes with the corresponding class
    property that returns its name as a string (or en Enum member that can function as a string). It is inherited by
    all objects' schemas and enables their deserialization from a DimcatConfig."""

    class Meta:
        ordered = True

    def get_attribute(self, obj: Any, attr: str, default: Any):
        if attr == "dtype":
            # the usual marshmallow.utils.get_value() tries to access attributes by subscripting first
            # this is a problem for the DimcatConfig which behaves like a dictionary where one of the options has
            # the key 'dtype' but, when serializing, it is the property 'dtype' that is relevant
            return obj.dtype
        return super().get_attribute(obj, attr, default)

    @classmethod
    @property
    def name(cls) -> str:
        """Qualified name of the schema, meaning it includes the name of the class that it is nested in."""
        return cls.__qualname__

    @mm.post_load()
    def init_object(self, data, **kwargs) -> DimcatObject:
        """Once the data has been loaded, create the corresponding object."""
        obj_name = data.pop("dtype")
        Constructor = get_class(obj_name)
        return Constructor(**data)

    @mm.pre_dump()
    def assert_type(self, obj, **kwargs):
        if not isinstance(obj, DimcatObject):
            raise mm.ValidationError(
                f"{self.name}: The object to be serialized needs to be a DimcatObject, not a {type(obj)!r}."
            )
        return obj

    @mm.post_dump()
    def validate_dump(self, data, **kwargs):
        """Make sure to never return invalid serialization data."""
        if "dtype" not in data:
            msg = (
                f"{self.name}: The serialized data doesn't have a 'dtype' field, meaning that DiMCAT would "
                f"not be able to deserialize it."
            )
            raise mm.ValidationError(msg)
        dtype_schema = get_schema(data["dtype"])
        report = dtype_schema.validate(data)
        if report:
            raise mm.ValidationError(
                f"Dump of {data['dtype']} created with a {self.name} could not be validated by "
                f"{dtype_schema.name} :\n{report}"
            )
        return dict(data)

    def __repr__(self):
        return f"{self.name}(many={self.many})"

    def __getattr__(self, item):
        raise AttributeError(
            f"AttributeError: {self.name!r} object has no attribute {item!r}"
        )


class DimcatObject(ABC):
    """All DiMCAT classes derive from DimcatObject, except for the nested Schema(DimcatSchema) class that they define or
    inherit."""

    _enum_type: ClassVar[Type[Enum]] = None
    """If a class specifies an Enum, its 'dtype' property returns the Enum member corresponding to its 'name'."""
    _registry: ClassVar[Dict[str, Type[DimcatObject]]] = {}
    """Registry of all subclasses (but not their corresponding Schema classes)."""

    class Schema(DimcatSchema):
        pass

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

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @property
    def dtype(cls) -> str | Enum:
        """Name of the class as enum member (if cls._enum_type is define, string otherwise)."""
        if cls._enum_type is None:
            return cls.name
        return cls._enum_type(cls.name)

    def filename_factory(self):
        return self.name

    @classmethod
    @property
    def schema(cls):
        """Returns the (instantiated) DimcatSchema singleton object for this class."""
        return get_schema(cls.dtype)

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
        """Serialize object to file.

        Args:
            filepath: Path to the text file to (over)write.
            indent: Prettify the JSON layout. Default indentation: 2 spaces
            **kwargs: Keyword arguments passed to :meth:`json.dumps`.

        Returns:

        """
        as_dict = self.to_dict()
        as_dict.update(**kwargs)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(as_dict, f, indent=indent, **kwargs)

    @classmethod
    def from_dict(cls, options, **kwargs) -> Self:
        options = dict(options, **kwargs)
        if "dtype" not in options:
            logger.debug(f"Added option {{'dtype': {cls.name}}}.")
            options["dtype"] = cls.name
        elif options["dtype"] != cls.name:
            logger.warning(
                f"Key 'dtype' was updated from {options['dtype']} to {cls.name}."
            )
            options["dtype"] = cls.name
        try:
            return cls.schema.load(options)
        except ValidationError as e:
            msg = f"Could not instantiate {cls.name} because {cls.schema.name}, failed to validate the options:\n{e}"
            raise ValidationError(msg)

    @classmethod
    def from_config(cls, config: DimcatConfig, **kwargs):
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


class DimcatConfig(MutableMapping, DimcatObject):
    """Behaves like a dictionary but accepts only keys and values that are valid under the Schema of the DimcatObject
    specified under the key 'dtype'. Every DimcatConfig needs to have a 'dtype' key that is the name of a DimcatObject
    and can specify zero or more additional key-value pairs that can be used to instantiate the described object.

    When dealing with a DimcatConfig you need to be aware that the 'dtype' and 'options' can been different things
    when used as keys as opposed to attributes (``DC`` represents a DimcatConfig):

    - ``DC['dtype']`` is the name of the described DimcatObject (equivalent to ``DC.options_dtype``)
    - ``DC.dtype`` returns the class name "DimcatConfig", according to all DimcatObjects' default behaviour
    - ``DC.options`` returns the key-value pairs wrapped by this config, which includes at least the 'dtype' key
    - ``DC['options']`` is the value of the 'options' option, which exists only if it is part of the described
      object's schema, for example if the described object is a :class:`DimcatConfig` itself.

    Examples:

        >>> from dimcat import DimcatConfig, DimcatObject
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

    class Schema(DimcatObject.Schema):
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

    @property
    def options_dtype(self):
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

    def validate(self, partial=False) -> Dict[str, List[str]]:
        """Validates the current status of the config in terms of ability to create an object. Empty dict == valid."""
        return self.options_schema.validate(self._options, many=False, partial=partial)

    def __getitem__(self, key):
        return self._options[key]

    def __delitem__(self, key):
        if key == "dtype":
            raise ValueError("Cannot remove key 'dtype' from DimcatConfig.")
        del self._options[key]

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

    def __iter__(self):
        return iter(self._options)

    def __len__(self):
        return len(self._options)

    def __repr__(self):
        return f"{self.name}({pformat(self._options)})"

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


class Data(DimcatObject):
    """
    This base class unites all classes containing data in some way or another.
    """

    class Schema(DimcatObject.Schema):
        # basedir = fields.String(required=True)
        pass


class PipelineStep(DimcatObject):
    """
    This abstract base class unites all classes able to transform some data in a pre-defined way.

    The initializer will set some parameters of the transformation, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.


    """

    class Schema(DimcatObject.Schema):
        pass

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns:
            True if the passed data is eligible.
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def process(self, data: Data) -> Data:
        """
        Perform a transformation on an input Data object. This should never alter the
        Data or its properties in place, instead returning a copy or view of the input.

        Args:
            data: The data to be transformed. This should not be altered in place.

        Returns:
            A copy of the input Data, potentially transformed in some way defined by this PipelineStep.
        """
        return data

    @overload
    def process_data(self, data: Data) -> Data:
        ...

    @overload
    def process_data(self, data: Iterable[Data]) -> List[Data]:
        ...

    def process_data(
        self, data: Union[Data, Iterable[Data]]
    ) -> Union[Data, List[Data]]:
        """Same as process(), with the difference that an Iterable is accepted."""
        if isinstance(data, Data):
            return self.process(data)
        return [self.process(d) for d in data]


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
