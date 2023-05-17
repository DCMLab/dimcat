from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC
from enum import Enum, IntEnum, auto
from functools import cache
from typing import (
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)
from zipfile import ZipFile

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from marshmallow import ValidationError, fields
from typing_extensions import Self

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
    SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = pd.DataFrame
    SomeSeries: TypeAlias = pd.Series

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=SomeDataframe)
S = TypeVar("S", bound=SomeSeries)

NEVER_STORE_UNVALIDATED_DATA = (
    False  # allows for skipping mandatory validations during development
)


def get_default_basepath():
    return os.getcwd()


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
    def validate_dump(self, data, **kwargs):
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

    def __init__(self):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.name] = cls

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
    """Behaves like a dictionary but accepts only keys and values that are valid under the Schema of the DimcatObject
    specified under the key 'dtype'.
    """

    def __init__(self, options=(), **kwargs):
        self._config = dict(options, **kwargs)
        if "dtype" not in self._config:
            raise ValueError(
                "DimcatConfig requires a 'dtype' key that needs to be the name of a DimcatObject."
            )
        dtype = self._config["dtype"]
        if isinstance(dtype, str):
            pass
        elif isinstance(dtype, DimcatObject) or issubclass(dtype, DimcatObject):
            dtype_str = dtype.name
            self._config["dtype"] = dtype_str
        else:
            raise ValueError(
                f"{dtype!r} is not the name of a DimcatObject, needed to instantiate a Config."
            )
        report = self.validate()
        if report:
            raise ValidationError(
                f"Cannot instantiate DimcatConfig with dtype={dtype!r} and invalid options:"
                f"\n{report}"
            )

    @property
    def dtype(self):
        return self._config["dtype"]

    @property
    def schema(self):
        """Returns the (instantiated) DimcatSchema singleton object for the class this Config describes."""
        return get_schema(self.dtype)

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
            raise ValidationError(
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
    This base class unites all classes containing data in some way or another.
    """

    class Schema(DimcatSchema):
        # basedir = fields.String(required=True)
        pass


class PipelineStep(DimcatObject):
    """
    This abstract base class unites all classes able to transform some data in a pre-defined way.

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


def check_rel_path(rel_path, basepath):
    if rel_path.startswith(".."):
        raise ValueError(
            f"{rel_path!r} points outside the basepath {basepath!r} which is not allowed."
        )
    if rel_path.startswith(f".{os.sep}") and len(rel_path) > 2:
        rel_path = rel_path[2:]
    return rel_path


def make_rel_path(path: str, start: str):
    """Like os.path.relpath() but ensures that path is contained within start."""
    rel_path = os.path.relpath(path, start)
    return check_rel_path(rel_path)


def get_base_and_file(abs_or_rel_path):
    if os.path.isabs(abs_or_rel_path):
        basepath, filepath = os.path.split(abs_or_rel_path)
    else:
        basepath = get_default_basepath()
        filepath = abs_or_rel_path
    return basepath, filepath


def make_tsv_resource() -> fl.Resource:
    """Returns a frictionless.Resource with the default properties of a TSV file stored to disk."""
    tsv_dialect = fl.Dialect.from_options(
        {
            "csv": {
                "delimiter": "\t",
            }
        }
    )
    options = {
        "scheme": "file",
        "mediatype": "text/tsv",
        "encoding": "utf-8",
        "dialect": tsv_dialect,
    }
    resource = fl.Resource(**options)
    resource.type = "table"
    return resource


def replace_ext(filepath, new_ext):
    file, _ = os.path.splitext(filepath)
    if file.split(".")[-1] in ("resource", "datapackage"):
        file = ".".join(file.split(".")[:-1])
    if new_ext[0] != ".":
        new_ext = "." + new_ext
    return file + new_ext


class ResourceStatus(IntEnum):
    """Expresses the status of a DimcatResource with respect to it being described, valid, and serialized to disk.
    Value 0 corresponds to empty (no data loaded). All following values have increasingly higher values following the
    logic "the higher the value, the closer the resource to its final status, SERIALIZED.
    That enables checking for a minimal status, e.g. ``if status >= ResourceStatus.VALIDATED``. This implies
    that every status includes all lower steps.
    """

    EMPTY = 0
    SCHEMA = auto()  # column_schema available but no dataframe has been loaded
    DATAFRAME = (
        auto()
    )  # dataframe available in memory but not yet validated against the column_schema
    VALIDATED = auto()  # validated dataframe available in memory
    SERIALIZED = (
        auto()
    )  # dataframe serialized to disk but not its descriptor -> can be changed or overwritten
    FROZEN = (
        auto()
    )  # descriptor pointing to the serialized dataframe has been written -> must be altered together


class DimcatResource(Generic[D], Data):
    """Data object wrapping a dataframe. The dataframe's metadata are stored as a :obj:`frictionless.Resource`, that
    can be used for serialization and (lazy) deserialization.
    """

    class Schema(Data.Schema):
        resource = fields.Method(
            serialize="get_resource_descriptor", deserialize="load_descriptor"
        )

        def get_resource_descriptor(self, obj: DimcatResource):
            if ResourceStatus.SCHEMA < obj.status < ResourceStatus.SERIALIZED:
                logger.info(
                    f"This {self.name} needs to be stored to disk to be expressed as restorable config."
                )
                if obj.status <= ResourceStatus.VALIDATED:
                    obj.store_dataframe()
            descriptor = obj._resource.to_descriptor()
            return descriptor

        def load_descriptor(self, data):
            return fl.Resource.from_descriptor(data)

    def __init__(
        self,
        df: Optional[D] = None,
        resource_name: Optional[str] = None,
        resource: Optional[fl.Resource | str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """

        Args:
            df: DataFrame that this object wraps.
            resource_name:
                Name of the resource. Used as filename if the resource is stored to a ZIP file. Defaults to
                :meth:`filename_factory`.
            resource: An existing :obj:`frictionless.Resource` or a file path resolving to a resource descriptor.
            column_schema:
                If you don't pass a schema or a path or URL to one, frictionless will try to infer it. However,
                this often leads to validation errors.
            basepath:
                The absolute path on the local machine, relative to which the resource will be described when written
                to disk. If not specified, it will default to

                * the current working directory if no ``filepath`` is given or the given filepath is relative
                * the ``filepath``'s directory if the filepath is absolute

            filepath:
                The path to the existing or future tabular resource on physical disk.

                * If it is a relative path, it will be appended to the ``basepath``.
                * If it is an absolute path, the directory will be used as ``basepath``, unless a basepath is specified,
                  in which case filepath must be contained in it, so that it can be expressed relatively to it.

            validate:
                By default, the DimcatResource will not be instantiated if the schema validation fails. Set to
                False if you want to skip the validation.
        """
        super().__init__()
        self._resource: fl.Resource = make_tsv_resource()
        self._status = ResourceStatus.EMPTY
        self._df: D = None
        if resource_name is None:
            self.resource_name = self.filename_factory()
        else:
            self.resource_name = resource_name
        if resource is not None:
            if isinstance(resource, str):
                self._resource = fl.Resource(resource)
                self.basepath = os.path.dirname(resource)
                assert self.normpath is not None and os.path.isfile(self.normpath), (
                    f"Resource does not point to an existing file "
                    f"(basepath: {self.basepath}):\n{self._resource}"
                )
                self._status = ResourceStatus.FROZEN
            else:
                self._resource = resource
        if resource_name is not None:
            self._resource.name = resource_name
        if column_schema is not None:
            self.column_schema = column_schema
        if df is not None:
            self.df = df
        if basepath is not None:
            self.basepath = basepath
        if filepath is not None:
            self.filepath = filepath
        if not self.is_frozen and self.is_serialized:
            self._status = ResourceStatus.SERIALIZED
        if validate and self.status == ResourceStatus.DATAFRAME:
            _ = self.validate_data(interrupt=NEVER_STORE_UNVALIDATED_DATA)

    @property
    def is_serialized(self) -> bool:
        return self.normpath is not None and os.path.isfile(self.normpath)

    @property
    def resource(self):
        return self._resource

    @property
    def status(self) -> ResourceStatus:
        return self._status

    @property
    def is_frozen(self) -> bool:
        """Whether or not this resource can be modified."""
        return self.status >= ResourceStatus.FROZEN

    @property
    def df(self) -> D:
        if self._df is not None:
            return self._df
        if self.is_frozen:
            return self.get_dataframe()

    @df.setter
    def df(self, df: D):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set dataframe on a resource whose valid descriptor has been written to disk."
            )
        if self._df is not None:
            raise RuntimeError("This resource already includes a dataframe.")
        self._df = df
        if not self.column_schema.fields:
            self.column_schema = fl.Schema.describe(df)
        self._status = ResourceStatus.DATAFRAME

    @property
    def basepath(self):
        return self._resource.basepath

    @basepath.setter
    def basepath(self, basepath):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set basepath on a resource whose valid descriptor has been written to disk."
            )
        assert os.path.isdir(
            basepath
        ), f"Basepath {basepath!r} is not an existing directory."
        if self.filepath is not None and os.path.isabs(self.filepath):
            rel_path = make_rel_path(self.filepath, basepath)
            self.filepath = rel_path
            logger.info(
                f"Absolute filepath became {rel_path} when setting basepath {basepath}"
            )
        self._resource.basepath = basepath

    @property
    def resource_name(self):
        return self._resource.name

    @resource_name.setter
    def resource_name(self, name: str):
        name_lower = name.lower()
        if not re.match(FRICTIONLESS_NAME_PATTERN, name_lower):
            raise ValueError(
                f"Name must be lowercase and work as filename: {name_lower!r}"
            )
        self._resource.name = name_lower
        if self.filepath is None:
            self.filepath = name_lower + ".tsv"

    @property
    def normpath(self):
        """Absolute path to the serialized or future tabular file."""
        n_specified = sum(path is not None for path in (self.basepath, self.filepath))
        if n_specified == 0:
            return
        if n_specified == 1:
            if self.filepath is None:
                return
            if os.path.isabs(self.filepath):
                return self.filepath
        if n_specified == 2:
            return os.path.normpath(os.path.join(self.basepath, self.filepath))

    @property
    def filepath(self):
        return self._resource.path

    @filepath.setter
    def filepath(self, filepath):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set filepath on a resource whose valid descriptor has been written to disk."
            )
        if os.path.isabs(filepath):
            if self.basepath is None:
                basepath, filepath = os.path.split(filepath)
                self.basepath = basepath
                logger.info(
                    f"Filepath split into basepath {self.basepath} and filepath {filepath}"
                )
            else:
                filepath = make_rel_path(filepath, self.basepath)
        else:
            if self.basepath is None:
                self.basepath = os.getcwd()
                logger.info(
                    f"Basepath set to current working directory {self.basepath}"
                )
            filepath = check_rel_path(filepath, self.basepath)
        self._resource.path = filepath

    @property
    def column_schema(self):
        default_na_values = ["<NA>"]
        for na in default_na_values:
            if na not in self._resource.schema.missing_values:
                self._resource.schema.missing_values.append(na)
        return self._resource.schema

    @column_schema.setter
    def column_schema(self, new_schema: fl.Schema):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set schema on a resource whose valid descriptor has been written to disk."
            )
        self._resource.schema = new_schema
        if self.status < ResourceStatus.SCHEMA:
            self._status = ResourceStatus.SCHEMA
        elif self.status >= ResourceStatus.VALIDATED:
            self._status = ResourceStatus.DATAFRAME

    def __str__(self):
        return str(self._resource)

    def __repr__(self):
        return repr(self._resource)

    def store_dataframe(
        self,
        name: Optional[str] = None,
        validate: bool = True,
    ):
        if name is None and self.is_frozen:
            raise RuntimeError(
                f"This {self.name} was originally read from disk and therefore is not being stored."
            )
        if self.status < ResourceStatus.DATAFRAME:
            raise RuntimeError(f"This {self.name} does not contain a dataframe.")
        if validate and self.status < ResourceStatus.VALIDATED:
            _ = self.validate_data(interrupt=NEVER_STORE_UNVALIDATED_DATA)

        if name is not None:
            filepath = name
        elif self.filepath is not None:
            filepath = self.filepath
        else:
            filepath = self.filename_factory()
        if not filepath.endswith(".tsv"):
            filepath += ".tsv"
        self.filepath = filepath
        if self.basepath is None:
            self.basepath = os.getcwd()
        full_path = os.path.normpath(os.path.join(self.basepath, filepath))
        if os.path.isfile(full_path):
            raise RuntimeError(
                f"File exists already on disk and will not be overwritten: {full_path}"
            )
        ms3.write_tsv(self.df, full_path)
        self._status = ResourceStatus.SERIALIZED

    @cache
    def get_dataframe(
        self, wrapped=True
    ) -> Union[DimcatResource[pd.DataFrame], pd.DataFrame]:
        r = self._resource
        if self._resource.path is None:
            raise ValidationError(
                "The resource does not refer to a file path and cannot be restored."
            )
        if r.normpath.endswith(".zip") or r.compression == "zip":
            zip_file_handler = ZipFile(r.normpath)
            df = ms3.load_tsv(zip_file_handler.open(r.innerpath))
        else:
            df = ms3.load_tsv(r.normpath)
        if len(r.schema.primary_key) > 0:
            df = df.set_index(r.schema.primary_key)
        if wrapped:
            return DimcatResource(df=df)
        return df

    @property
    def is_valid(self) -> bool:
        if self.status < ResourceStatus.DATAFRAME:
            return False
        if self.status >= ResourceStatus.VALIDATED:
            return True
        report = self.validate_data()
        if report is not None:
            return report.valid

    def validate_data(self, interrupt: bool = False) -> Optional[fl.Report]:
        if self.status < ResourceStatus.DATAFRAME:
            logger.info("Nothing to validate.")
            return
        if self.is_frozen:
            report = self._resource.validate()
        else:
            tmp_resource = fl.Resource(self.df)
            tmp_resource.schema = self.column_schema
            report = tmp_resource.validate()
        if report.valid:
            if self.status < ResourceStatus.VALIDATED:
                self._status = ResourceStatus.VALIDATED
        elif interrupt:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise ValidationError("\n".join(errors))
        return report

    def __getattr__(self, item):
        """Enables using DimcatResource just like the wrapped DataFrame."""
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
