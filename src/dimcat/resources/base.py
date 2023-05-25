from __future__ import annotations

import logging
import os
import re
from enum import IntEnum, auto
from functools import cache
from typing import Generic, List, Optional, TypeAlias, TypeVar, Union
from zipfile import ZipFile

import frictionless as fl
import ms3
import pandas as pd
from dimcat.base import Data
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from marshmallow import ValidationError, fields

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
    False  # allows for skipping mandatory validations; set to True for production
)

# region helper functions


def get_default_basepath():
    return os.getcwd()


def infer_schema_from_df(df: SomeDataframe) -> fl.Schema:
    """Infer a frictionless.Schema from a dataframe."""
    return fl.Schema.describe(df)


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


# endregion helper functions


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
    ON_DISK_AND_LOADED = (
        auto()
    )  # descriptor pointing to the serialized dataframe has been written -> must be altered together
    ON_DISK_NOT_LOADED = (
        auto()
    )  # like ON_DISK_AND_LOADED but the dataframe has not been loaded into memory


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
                The absolute path on the local file system, relative to which the resource will be described when
                written to disk. If not specified, it will default to

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
        self.descriptor_path: Optional[str] = None

        if resource is not None:
            if resource_name is None:
                self.resource_name = self.filename_factory()
            else:
                # this purposefully happens twice, the first time to allow for computing paths when setting a resource
                # and the second time to overwrite the name of the given resource
                self.resource_name = resource_name
            if isinstance(resource, str):
                self._resource = fl.Resource(resource)
                self.basepath = os.path.dirname(resource)
                assert self.normpath is not None and os.path.isfile(self.normpath), (
                    f"Resource does not point to an existing file "
                    f"(basepath: {self.basepath}):\n{self._resource}"
                )
                self.descriptor_path = self.normpath
                self._status = ResourceStatus.ON_DISK_NOT_LOADED
            elif isinstance(resource, fl.Resource):
                self._resource = resource
                if resource.scheme == "file":
                    self._status = ResourceStatus.ON_DISK_NOT_LOADED
            else:
                raise TypeError(
                    f"Expected a path or a frictionless resource, got {type(resource)}"
                )

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
            _ = self.validate_data(raise_exception=NEVER_STORE_UNVALIDATED_DATA)

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

    @property
    def df(self) -> D:
        if self._df is not None:
            return self._df
        if self.is_frozen:
            return self.get_dataframe()
        raise RuntimeError(f"No dataframe accessible for this {self.name}:\n{self}")

    @df.setter
    def df(self, df: D):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set dataframe on a resource whose valid descriptor has been written to disk."
            )
        if self.is_loaded:
            raise RuntimeError("This resource already includes a dataframe.")
        self._df = df
        if not self.column_schema.fields:
            self.column_schema = infer_schema_from_df(df)
        self._status = ResourceStatus.DATAFRAME

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
                self.basepath = get_default_basepath()
                logger.info(
                    f"Basepath set to current working directory {self.basepath}"
                )
            filepath = check_rel_path(filepath, self.basepath)
        self._resource.path = filepath

    @property
    def is_frozen(self) -> bool:
        """Whether the resource is frozen (i.e. its valid descriptor has been written to disk) or not."""
        return self.status >= ResourceStatus.ON_DISK_AND_LOADED

    @property
    def is_loaded(self) -> bool:
        return (
            ResourceStatus.DATAFRAME <= self.status < ResourceStatus.ON_DISK_NOT_LOADED
        )

    @property
    def is_serialized(self) -> bool:
        return self.normpath is not None and os.path.isfile(self.normpath)

    @property
    def is_valid(self) -> bool:
        if self.status < ResourceStatus.DATAFRAME:
            return False
        if self.status >= ResourceStatus.VALIDATED:
            return True
        report = self.validate_data()
        if report is not None:
            return report.valid

    @property
    def normpath(self) -> Optional[str]:
        """Absolute path to the serialized or future tabular file."""
        if self.descriptor_path is not None:
            return self.descriptor_path
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
    def resource(self) -> fl.Resource:
        return self._resource

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
    def status(self) -> ResourceStatus:
        if self._status == ResourceStatus.EMPTY and self._resource.schema.fields:
            self._status = ResourceStatus.SCHEMA
        return self._status

    @cache
    def get_dataframe(self, wrapped=True) -> Union[DimcatResource[D], D]:
        """
        Load the dataframe from disk based on the descriptor's normpath.

        Args:
            wrapped: By default only the underlying dataframe. Set to True if you want a DimcatResource wrapper.

        Returns:
            The dataframe or DimcatResource.
        """
        r = self._resource
        if self.normpath is None:
            raise ValidationError(
                f"The resource does not refer to a file path and cannot be loaded.\n"
                f"basepath: {self.basepath}, filepath: {self.filepath}"
            )
        if self.normpath.endswith(".zip") or r.compression == "zip":
            zip_file_handler = ZipFile(self.normpath)
            df = ms3.load_tsv(zip_file_handler.open(r.innerpath))
        else:
            df = ms3.load_tsv(self.normpath)
        if len(r.schema.primary_key) > 0:
            df = df.set_index(r.schema.primary_key)
        if self.status == ResourceStatus.ON_DISK_NOT_LOADED:
            self._status = ResourceStatus.ON_DISK_AND_LOADED
        if wrapped:
            return DimcatResource(df=df)
        return df

    def load(self, force_reload: bool = False):
        """Tries to load the data from disk into RAM. If successful, the .is_loaded property will be True."""
        if not self.is_loaded or force_reload:
            _ = self.df

    def store_dataframe(
        self,
        name: Optional[str] = None,
        validate: bool = True,
    ):
        if self.is_frozen:
            raise RuntimeError(
                f"This {self.name} was originally read from disk and therefore is not being stored."
            )
        if self.status < ResourceStatus.DATAFRAME:
            raise RuntimeError(f"This {self.name} does not contain a dataframe.")
        if validate and self.status < ResourceStatus.VALIDATED:
            _ = self.validate_data(raise_exception=NEVER_STORE_UNVALIDATED_DATA)

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
            self.basepath = get_default_basepath()
        full_path = os.path.normpath(os.path.join(self.basepath, filepath))
        if os.path.isfile(full_path):
            raise RuntimeError(
                f"File exists already on disk and will not be overwritten: {full_path}"
            )
        ms3.write_tsv(self.df, full_path)
        self._status = ResourceStatus.SERIALIZED

    def validate_data(self, raise_exception: bool = False) -> Optional[fl.Report]:
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
        elif raise_exception:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise fl.FrictionlessException("\n".join(errors))
        return report

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = super().__dir__()
        elements.extend(dir(self.df))
        return sorted(elements)

    def __getattr__(self, item):
        """Enables using DimcatResource just like the wrapped DataFrame."""
        msg = f"{self.name!r} object ({self.status!r}) has no attribute {item!r}."
        if not self.is_loaded:
            msg += " Try again after loading the dataframe into memory."
            raise AttributeError(msg)
        try:
            return getattr(self.df, item)
        except AttributeError:
            raise AttributeError(msg)

    def __getitem__(self, item):
        try:
            return self.df[item]
        except Exception:
            raise KeyError(item)

    def __len__(self) -> int:
        return len(self.df.index)

    def __repr__(self):
        return repr(self._resource)

    def __str__(self):
        return str(self._resource)

    def __hash__(self):
        return id(self)
