from __future__ import annotations

import json
from abc import ABC
from enum import Enum
from functools import cache
from typing import (
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    MutableMapping,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import marshmallow as mm
import pandas as pd
from typing_extensions import Self

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
    SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = pd.DataFrame
    SomeSeries: TypeAlias = pd.Series

D = TypeVar("D", bound=SomeDataframe)
S = TypeVar("S", bound=SomeSeries)


class DimcatSchema(mm.Schema):
    """
    The base class of all Schema() classes that are defined or inherited as nested classes
    for all :class:`DimcatObjects <DimcatObject>`. This class holds the logic for serializing/deserializing DiMCAT
    objects.
    """

    dtype = mm.fields.String()
    """This field specifies the class of the serialized object. Every DimcatObject comes with the corresponding class
    property that returns its name as a string (or en Enum member that can function as a string)."""

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

    @mm.post_dump()
    def validate_dump(self, data, **kwargs) -> None:
        """Make sure to never return invalid serialization data."""
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


class DimcatObject(ABC):
    """All DiMCAT classes derive from DimcatObject, except for the nested Schema(DimcatSchema) class that they define or
    inherit."""

    _enum_type: ClassVar[Type[Enum]] = None
    """If a class specifies an Enum, its 'dtype' property returns the Enum member corresponding to its 'name'."""
    _registry: ClassVar[Dict[str, Type[DimcatObject]]] = {}
    """Registry of all subclasses (but not their corresponding Schema classes)."""

    class Schema(DimcatSchema):
        pass

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

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

    @classmethod
    @property
    def schema(cls):
        """Returns the (instantiated) DimcatSchema singleton object for this class."""
        return get_schema(cls.name)

    def to_dict(self) -> dict:
        return self.schema.dump(self)

    def to_config(self) -> DimcatConfig:
        return DimcatConfig(self.to_dict())

    def to_json(self) -> str:
        return self.schema.dumps(self)

    @classmethod
    def from_dict(cls, config, **kwargs) -> Self:
        config = dict(config, **kwargs)
        config["dtype"] = cls.name
        return cls.schema.load(config)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls.from_dict(config, **kwargs)

    @classmethod
    def from_json(cls, config: str, **kwargs) -> Self:
        json_dict = json.loads(config)
        return cls.from_dict(json_dict, **kwargs)


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


class Data(DimcatObject):
    """
    This abstract base class unites all classes containing data in some way or another.
    All subclasses are dynamically collected in the class variable :attr:`_registry`.
    """

    pass


class PipelineStep(DimcatObject):
    """
    This abstract base class unites all classes able to transform some data in a pre-defined way.
    All subclasses are dynamically collected in the class variable :attr:`_registry`.

    The initializer will set some parameters of the transformation, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.


    """

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns:
            True if the passed data is eligible.
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def filename_factory(self):
        return self.name

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


class WrappedSeries(Generic[S], Data):
    """Wrapper around a Series.
    Can be used like the wrapped series but subclasses may provide additional functionality.
    """

    def __init__(self, series: S, **kwargs):
        super().__init__(**kwargs)
        self._series: D = series
        """The wrapped Series object."""

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        raise RuntimeError(
            f"Cannot assign to the field series. Use {self.name}.from_series() to create a new object."
        )

    @classmethod
    def from_series(cls, series: S, **kwargs):
        """Subclasses can implement transformational logic."""
        instance = cls(series=series, **kwargs)
        return instance

    def __getitem__(self, int_or_slice_or_mask):
        if isinstance(int_or_slice_or_mask, (int, slice)):
            return self.series.iloc[int_or_slice_or_mask]
        if isinstance(int_or_slice_or_mask, pd.Series):
            return self.series[int_or_slice_or_mask]
        raise KeyError(f"{self.name} cannot be subscripted with {int_or_slice_or_mask}")

    def __getattr__(self, item):
        """Enable using IndexSequence like a Series."""
        return getattr(self.series, item)

    def __len__(self) -> int:
        return len(self.series.index)


class WrappedDataframe(Generic[D], Data):
    """Wrapper around a DataFrame.
    Can be used like the wrapped dataframe but subclasses may provide additional functionality.
    """

    def __init__(self, df: D, **kwargs):
        super().__init__(**kwargs)
        self._df: D = df
        """The wrapped Dataframe object."""

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        raise RuntimeError(
            f"Cannot assign to the field df. Use {self.name}.from_df() to create a new object."
        )

    @classmethod
    def from_df(cls, df: D, **kwargs) -> Self:
        """Subclasses can implement transformational logic."""
        instance = cls(df=df, **kwargs)
        return instance

    def get_column(self, column_name: str) -> WrappedSeries:
        column = self.df.loc[:, column_name]
        return WrappedSeries(column)

    def __getattr__(self, item):
        """Enables using WrappedDataframe like the wrapped DataFrame."""
        return getattr(self.df, item)

    def __getitem__(self, item):
        return self.df[item]

    def __len__(self) -> int:
        return len(self.df.index)

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = super().__dir__()
        elements.extend(dir(self.df))
        return sorted(elements)


@cache
def get_class(name) -> Type[DimcatObject]:
    if name == "DimcatObject":
        # this is the only object that's not in the registry
        return DimcatObject
    return DimcatObject._registry[name]


@cache
def is_dimcat_class(name) -> bool:
    return name in DimcatObject._registry


@cache
def get_schema(name, init=True):
    """Caches the intialized schema for each class. Pass init=False to retrieve the schema constructor."""
    dc_class = get_class(name)
    dc_schema = dc_class.Schema
    if init:
        return dc_schema()
    return dc_schema
