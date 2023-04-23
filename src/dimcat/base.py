from __future__ import annotations

from abc import ABC
from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import pandas as pd
from typing_extensions import Self

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = Union[pd.DataFrame]

D = TypeVar("D", bound=SomeDataframe)


@dataclass()
class Configuration(ABC):
    _configured_class: ClassVar[str]

    @classmethod
    def from_dataclass(cls, config: Configuration, **kwargs) -> Self:
        """This class methods copies the fields it needs from another config-like dataclass."""
        init_args = cls.dict_from_dataclass(config, **kwargs)
        return cls(**init_args)

    @classmethod
    def from_dict(cls, config: dict, **kwargs) -> Self:
        """This class methods copies the fields it needs from another config-like dataclass."""
        if not isinstance(config, dict):
            raise TypeError(
                f"Expected a dictionary, received a {type(config)!r} instead."
            )
        config = dict(config)
        config.update(kwargs)
        field_names = [field.name for field in fields(cls) if field.init]
        init_args = {key: value for key, value in config.items() if key in field_names}
        return cls(**init_args)

    @classmethod
    def dict_from_dataclass(cls, config: Configuration, **kwargs) -> Dict:
        """This class methods copies the fields it needs from another config-like dataclass."""
        init_args: Dict[str, Any] = {}
        field_names = []
        for config_field in fields(cls):
            if not config_field.init:
                continue
            field_name = config_field.name
            field_names.append(config_field.name)
            if not hasattr(config, field_name):
                continue
            init_args[field_name] = getattr(config, field_name)
        init_args.update(kwargs)
        return init_args

    def __eq__(self, other):
        return astuple(self) == astuple(other)


class DimcatObject(ABC):
    """All DiMCAT classes derive from DimcatObject and can be"""

    _config_type: ClassVar[Type[Configuration]] = Configuration
    _registry: ClassVar[Dict[str, Type]] = {}
    """Register of all subclasses."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = self._config_type.from_dict(kwargs)
        else:
            config = self._config_type.from_dataclass(config, **kwargs)
        self._config: Configuration = config

    @property
    def config(self) -> Configuration:
        return self._config

    @classmethod
    @property
    def dtype(cls) -> Union[Enum, str]:
        """Name of the class as enum member (if cls._enum_type is define, string otherwise)."""
        if hasattr(cls, "_enum_type"):
            return cls._enum_type(cls.name)
        return cls.name

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def from_config(cls, config: Configuration) -> DimcatObject:
        type_str = config._configured_class
        constructor = cls._registry[type_str]
        return constructor(config)


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


@dataclass(frozen=True)
class WrappedDataframe(Generic[D]):
    """Wrapper around a DataFrame.
    Can be used like the wrapped dataframe but subclasses provide additional functionality.
    """

    df: D
    """The wrapped Dataframe object."""

    @classmethod
    def from_df(cls, df: D, **kwargs) -> Self:
        """Subclasses can implement transformational logic."""
        instance = cls(df=df, **kwargs)
        return instance

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
