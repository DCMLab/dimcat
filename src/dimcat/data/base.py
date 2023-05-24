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
    False  # allows for skipping mandatory validations during development
)

# region helper functions


def get_default_basepath():
    return os.getcwd()


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
        """Whether or not this resource can be modified."""
        return self.status >= ResourceStatus.FROZEN

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
    def resource(self):
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
        return self._status

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
            self.basepath = get_default_basepath()
        full_path = os.path.normpath(os.path.join(self.basepath, filepath))
        if os.path.isfile(full_path):
            raise RuntimeError(
                f"File exists already on disk and will not be overwritten: {full_path}"
            )
        ms3.write_tsv(self.df, full_path)
        self._status = ResourceStatus.SERIALIZED

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

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = super().__dir__()
        elements.extend(dir(self.df))
        return sorted(elements)

    def __getattr__(self, item):
        """Enables using DimcatResource just like the wrapped DataFrame."""
        return getattr(self.df, item)

    def __getitem__(self, item):
        return self.df[item]

    def __len__(self) -> int:
        return len(self.df.index)

    def __repr__(self):
        return repr(self._resource)

    def __str__(self):
        return str(self._resource)