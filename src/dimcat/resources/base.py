from __future__ import annotations

import logging
import os
import re
import zipfile
from enum import IntEnum, auto
from functools import cache
from pprint import pformat
from typing import (
    Collection,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)
from zipfile import ZipFile

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import NEVER_STORE_UNVALIDATED_DATA, Data, DimcatConfig, get_class
from dimcat.utils import replace_ext
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from typing_extensions import Self

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
    SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
    SomeIndex: TypeAlias = Union[pd.Index, mpd.Index]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = pd.DataFrame
    SomeSeries: TypeAlias = pd.Series
    SomeIndex: TypeAlias = pd.Index

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=SomeDataframe)
S = TypeVar("S", bound=SomeSeries)
IX = TypeVar("IX", bound=SomeIndex)


# region helper functions


def check_file_path(
    filepath: str,
    extensions: Optional[str | Collection[str]] = None,
    must_exist: bool = True,
) -> str:
    """Checks that the filepath exists and raises an exception otherwise (or if it doesn't have a valid extension).

    Args:
        filepath:
        extensions:
        must_exist: If True (default), raises FileNotFoundError if the file does not exist.

    Returns:
        The path turned into an absolute path.
    """
    path = ms3.resolve_dir(filepath)
    if must_exist and not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    if extensions is not None:
        if isinstance(extensions, str):
            extensions = [extensions]
        if not any(path.endswith(ext) for ext in extensions):
            plural = f"one of {extensions}" if len(extensions) > 1 else extensions[0]
            _, file_ext = os.path.splitext(path)
            raise ValueError(f"File {path} has extension {file_ext}, not {plural}.")
    return path


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


def make_tsv_resource(name: Optional[str] = None) -> fl.Resource:
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
        "format": "tsv",
        "mediatype": "text/tsv",
        "encoding": "utf-8",
        "dialect": tsv_dialect,
    }
    if name is not None:
        options["name"] = name
    resource = fl.Resource(**options)
    resource.type = "table"
    return resource


def fl_fields2pandas_params(fields: List[fl.Field]) -> Tuple[dict, dict]:
    """Convert frictionless Fields to pandas parameters 'dtype' and 'converters'."""
    dtype = {}
    converters = {}
    for field in fields:
        if not field.type:
            continue
        if field.type == "string":
            dtype[field.name] = "string"
        elif field.type == "integer":
            if field.required:
                dtype[field.name] = int
            else:
                dtype[field.name] = "Int64"
        elif field.type == "number":
            dtype[field.name] = float
        elif field.type == "boolean":
            dtype[field.name] = bool
        # missing (see https://specs.frictionlessdata.io/table-schema)
        # - object (i.e. JSON/ a dict)
        # - array (i.e. JSON/ a list)
        # - date (i.e. date without time)
        # - time (i.e. time without date)
        # - datetime (i.e. a date with a time)
        # - year (i.e. a year without a month and day)
        # - yearmonth (i.e. a year and month without a day)
        # - duration (i.e. a time duration)
        # - geopoint (i.e. a pair of lat/long coordinates)
        # - geojson (i.e. a GeoJSON object)
        else:
            raise ValueError(f"Unknown frictionless field type {field.type!r}.")
    return dtype, converters


def infer_piece_col_position(
    column_name: List[str],
    recognized_piece_columns: Iterable[str] = ("piece", "fname"),
) -> Optional[int]:
    """Infer the position of the piece column in a list of column names."""
    for name in recognized_piece_columns:
        try:
            return column_name.index(name)
        except ValueError:
            continue
    return


def get_existing_normpath(fl_resource) -> str:
    if fl_resource.normpath is None:
        if fl_resource.path is None:
            raise ValueError(f"Resource {fl_resource.name!r} has no path.")
        normpath = os.path.abspath(fl_resource.path)
    else:
        normpath = fl_resource.normpath
    if not os.path.isfile(normpath):
        raise FileNotFoundError(
            f"Normpath of resource {fl_resource.name!r} does not exist: {normpath}"
        )
    if not fl_resource.schema.fields:
        raise ValueError(f"Resource {fl_resource.name!r}'s column_schema is empty.")
    return normpath


def resolve_columns_argument(
    columns: Optional[int | str | List[int | str]],
    field_names: List[str],
) -> Optional[List[str]]:
    if columns is None:
        return
    result = []
    for str_or_int in columns:
        if isinstance(str_or_int, int):
            result.append(field_names[str_or_int])
        else:
            if str_or_int not in field_names:
                raise ValueError(
                    f"Column {str_or_int!r} not found in field names {field_names}."
                )
            result.append(str_or_int)
    result_set = set(result)
    if len(result_set) != len(result):
        raise ValueError(f"Duplicate column names in {columns}.")
    return result


def load_fl_resource(
    fl_resource: fl.Resource,
    index_col: Optional[int | str | List[int | str]] = None,
    usecols: Optional[Union[int, str, List[int | str]]] = None,
) -> SomeDataframe:
    normpath = get_existing_normpath(fl_resource)
    field_names = fl_resource.schema.field_names
    index_col_names = resolve_columns_argument(index_col, field_names)
    usecols_names = resolve_columns_argument(usecols, field_names)
    usecols_fields = [fl_resource.schema.get_field(name) for name in usecols_names]
    dtypes, converters = fl_fields2pandas_params(usecols_fields)
    if normpath.endswith(".zip"):
        if not fl_resource.innerpath:
            raise ValueError(
                f"Resource {fl_resource.name!r} is a zip file but has no innerpath."
            )
        zip_file_handler = ZipFile(normpath)
        file = zip_file_handler.open(fl_resource.innerpath)
    else:
        file = normpath
    try:
        dataframe = pd.read_csv(
            file, sep="\t", usecols=usecols_names, dtype=dtypes, converters=converters
        )
    except Exception:
        logger.warning(
            f"Error executing\n"
            f"pd.read_csv({file!r},\n"
            f"            sep='\\t',\n"
            f"            usecols={usecols_names!r},\n"
            f"            dtype={dtypes!r},\n"
            f"            converters={converters!r})"
        )
        raise
    if index_col_names:
        return dataframe.set_index(index_col_names)
    return dataframe


def load_index_from_fl_resource(
    fl_resource: fl.Resource,
    index_col: Optional[int | str | List[int | str]] = None,
    recognized_piece_columns: Iterable[str] = ("piece", "fname"),
) -> SomeIndex:
    """Load the index columns from a frictionless Resource.

    Args:
        fl_resource: The frictionless Resource to load the index columns from.
        index_col: The column(s) to use as index. If None, the primary key of the schema is used if it exists.
        recognized_piece_columns:
            If the loaded columns do not include 'piece' but one of the names specified here, the first column name
            of the iterable that is detected in the loaded columns will be renamed to 'piece'. Likewise, such a
            column would be used (and renamed) if ``index_col`` is not specified *and* the schema does not specify
            a primary key: in that case, the detected column and all columns left of it will used as index_col argument.
            The iterable should always start with "piece" -- otherwise the same mechanism will be applied for
            whatever first value specified.

    Returns:
        The specified or inferred index column(s) as a (Multi)Index object.

    Raises:
        FileNotFoundError: If the normpath of the resource does not exist.
        ValueError: If the resource doesn't yield a normpath or the index columns cannot be inferred from it
            based on the schema.
    """
    _ = get_existing_normpath(fl_resource)  # raises if normpath doesn't exist
    schema = fl_resource.schema
    if recognized_piece_columns:
        recognized_piece_columns = tuple(recognized_piece_columns)
        right_most_column = recognized_piece_columns[0]
    else:
        right_most_column = None
    if index_col is not None:
        if isinstance(index_col, (int, str)):
            index_col = [index_col]
    elif schema.primary_key:
        index_col = schema.primary_key
    elif right_most_column:
        logger.debug(
            f"Resource {fl_resource.name!r} has no primary key, trying to detect {right_most_column!r} column."
        )
        piece_col_position = infer_piece_col_position(
            schema.field_names,
            recognized_piece_columns=recognized_piece_columns,
        )
        if piece_col_position is None:
            raise ValueError(
                f"Resource {fl_resource.name!r} has no primary key and no {right_most_column!r} column "
                f"could be detected."
            )
        index_col = list(range(piece_col_position + 1))
    else:
        raise ValueError(
            f"Resource {fl_resource.name!r} has no primary key and neither index_col nor recognized_piece_columns "
            f"were specified."
        )
    dataframe = load_fl_resource(fl_resource, usecols=index_col)
    if right_most_column not in dataframe.columns:
        columns = dataframe.columns.to_list()
        piece_col_position = infer_piece_col_position(
            columns,
            recognized_piece_columns=recognized_piece_columns,
        )
        if piece_col_position is not None:
            dataframe = dataframe.rename(
                columns={columns[piece_col_position]: right_most_column}
            )
    return dataframe.set_index(dataframe.columns.to_list()).index


# endregion helper functions

# region DimcatIndex


class DimcatIndex(Generic[IX], Data):
    """A wrapper around a :obj:`pandas.MultiIndex` that provides additional functionality such as keeping track of
    index levels and default groupings.

    A MultiIndex essentially is a Sequence of tuples where each tuple identifies dataframe row and includes one value
    per index level. Each index level has a name and can be seen as in individual :obj:`pandas.Index`. One important
    type of DimcatIndex is the PieceIndex which is a unique MultiIndex (that is, each tuple is unique) and where the
    last (i.e. right-most) level is named `piece`.
    """

    @classmethod
    def from_resource(
        cls,
        resource: DimcatResource | fl.Resource,
        index_col: Optional[int | str | List[int | str]] = None,
    ) -> Self:
        """Create a DimcatIndex from a frictionless Resource."""
        if isinstance(resource, DimcatResource):
            if resource.status < ResourceStatus.DATAFRAME:
                return cls()
            if resource.is_loaded:
                return cls(resource.df.index)
            fl_resource = resource.resource
        elif isinstance(resource, fl.Resource):
            fl_resource = resource
        else:
            raise TypeError(
                f"Expected DimcatResource or frictionless.Resource, got {type(resource)!r}."
            )
        # load only the index columns from the serialized resource
        index = load_index_from_fl_resource(fl_resource, index_col=index_col)
        return cls(index)

    def __init__(self, index: Optional[IX] = None):
        if index is None:
            self._index = pd.Index()
        else:
            self._index = index.copy()

    def __getattr__(self, item):
        """Enables using DimcatIndex just like the wrapped Index object."""
        return getattr(self._index, item)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return str(self._index)

    @property
    def index(self) -> IX:
        return self._index


class PieceIndex(DimcatIndex):
    """A unique DimcatIndex where the last (i.e. right-most) level is named `piece`."""

    pass


# endregion DimcatIndex
# region DimcatResource


class ResourceStatus(IntEnum):
    """Expresses the status of a DimcatResource with respect to it being described, valid, and serialized to disk,
    with or without its descriptor file.

    +-----------------------+---------------+-------------------+-----------+
    | ResourceStatus        | is_serialized | descriptor_exists | is_loaded |
    +=======================+===============+===================+===========+
    | EMPTY                 | False         | ?                 | False     |
    +-----------------------+---------------+-------------------+-----------+
    | SCHEMA                | False         | ?                 | False     |
    +-----------------------+---------------+-------------------+-----------+
    | DATAFRAME             | False         | False             | True      |
    +-----------------------+---------------+-------------------+-----------+
    | VALIDATED             | False         | False             | True      |
    +-----------------------+---------------+-------------------+-----------+
    | SERIALIZED            | True          | False             | True      |
    +-----------------------+---------------+-------------------+-----------+
    | STANDALONE_LOADED     | True          | True              | True      |
    +-----------------------+---------------+-------------------+-----------+
    | PACKAGED_LOADED       | True          | True              | True      |
    +-----------------------+---------------+-------------------+-----------+
    | STANDALONE_NOT_LOADED | True          | True              | False     |
    +-----------------------+---------------+-------------------+-----------+
    | PACKAGED_NOT_LOADED   | True          | True              | False     |
    +-----------------------+---------------+-------------------+-----------+
    """

    EMPTY = 0
    SCHEMA = auto()  # column_schema available but no dataframe has been loaded
    DATAFRAME = (
        auto()
    )  # dataframe available in memory but not yet validated against the column_schema
    VALIDATED = auto()  # validated dataframe available in memory
    SERIALIZED = (
        auto()
    )  # dataframe serialized to disk but not its descriptor (shouldn't happen) -> can be changed or overwritten
    STANDALONE_LOADED = auto()
    PACKAGED_LOADED = auto()
    STANDALONE_NOT_LOADED = auto()
    PACKAGED_NOT_LOADED = auto()


class DimcatResource(Generic[D], Data):
    """Data object wrapping a dataframe. The dataframe's metadata are stored as a :obj:`frictionless.Resource`, that
    can be used for serialization and (lazy) deserialization.

    Every serialization of a DimcatResource (e.g. to store it as a config) requires that the dataframe was either
    originally read from disk or, otherwise, that it be stored to disk. The behaviour depends on whether the resource
    is part of a package or not.

    Standalone resource (rare case)
    -------------------------------

    If the resource is not part of a package, serializing it results in two files on disk:

    - the dataframe stored as ``<basepath>/<name>.tsv``
    - the frictionless descriptor ``<basepath>/<name>.resource.json``

    where ``<name>`` defaults to ``resource_name`` unless ``filepath`` is specified. The serialization has the shape

    .. code-block:: python

        {
            "dtype": "DimcatResource",
            "resource": "<name>.resource.json",
            "basepath": "<basepath>"
        }

    A standalone resource can be instantiated in the following ways:

    - ``DimcatResource()``: Creates an empty DimcatResource for setting the .df attribute later. If no ``basepath``
      is specified, the current working directory is used if the resource is to be serialized.
    - ``DimcatResource.from_descriptor(descriptor_path)``: The frictionless descriptor is loaded from disk.
      Its directory is used as ``basepath``. ``descriptor_path`` is expected to end in ``.resource.json``.
    - ``DimcatResource.from_dataframe(df=df, resource_name, basepath)``: Creates a new DimcatResource from a dataframe.
      If ``basepath`` is not specified, the current working directory is used if the resource is to be serialized.
    - ``DimcatResource.from_resource(resource=DimcatResource)``: Creates a DimcatResource from an existing one
      by copying the fields it specifies.

    Resource in a package (common case)
    -----------------------------------

    A DimcatResource knows that it is part of a package if its ``filepath`` ends on ``.zip``. In that case, the
    DimcatPackage will take care of the serialization and not store an individual resource descriptor.
    """

    @classmethod
    @property
    def pickle_schema(cls):
        """Returns the (instantiated) PickleSchema singleton object for this class. It is different from the 'noremal'
        Schema in that it stores the tabular data to disk and returns the path to its descriptor.
        """
        return get_pickle_schema(cls.dtype)

    @classmethod
    def from_descriptor(
        cls,
        descriptor_path: str,
        auto_validate: bool = False,
    ) -> Self:
        """Create a DimcatResource by loading its frictionless descriptor is loaded from disk.
        The descriptor's directory is used as ``basepath``. ``descriptor_path`` is expected to end in
        ``.resource.json``.

        Args:
            descriptor_path: Needs to be an absolute path and is expected to end in ``.resource.json``.
            auto_validate:
                By default, the DimcatResource will not be instantiated if the schema validation fails and the resource
                is re-validated if, for example, the :attr:`column_schema` changes. Set False to prevent validation.
        """
        return cls(resource=descriptor_path, auto_validate=auto_validate)

    @classmethod
    def from_dataframe(
        cls,
        df: D,
        resource_name: str,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> Self:
        """Create a DimcatResource from a dataframe, specifying its name and, optionally, at what path it is to be
        serialized.

        Args:
            df: Dataframe to create the resource from.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath:
                The absolute path on the local file system, relative to which the resource will be described when
                written to disk. If not specified, it will default to

                * the current working directory if no ``filepath`` is given or the given filepath is relative
                * the ``filepath``'s directory if the filepath is absolute

            filepath:
                The path to the existing or future tabular resource on physical disk.

                * If None it defaults to ``resource_name`` with the extension ``.tsv`` (which also serves as innerpath
                  in case ``filepath`` points to a ZIP file).
                * If it is a relative path, it will be appended to the ``basepath``.
                * If it is an absolute path, the directory will be used as ``basepath``, unless a basepath is specified,
                  in which case filepath must be contained in it, so that it can be expressed relatively to it.

            column_schema:
                If you don't pass a schema or a path or URL to one, frictionless will try to infer it. However,
                this often leads to validation errors.
            auto_validate:
                By default, the DimcatResource will not be instantiated if the schema validation fails and the resource
                is re-validated if, for example, the :attr:`column_schema` changes. Set False to prevent validation.
        """
        new_resource = cls(
            resource_name=resource_name,
            basepath=basepath,
            filepath=filepath,
            column_schema=column_schema,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
        )
        new_resource.df = df
        return new_resource

    @classmethod
    def from_dict(cls, options, **kwargs) -> Self:
        """Other than a config-like dict, this constructor also accepts a minimal descriptor dict resulting from
        serializing an empty DimcatResource."""
        if "dtype" not in options and "resource" not in options:
            return cls(**options, **kwargs)
        return super().from_dict(options, **kwargs)

    @classmethod
    def from_resource(
        cls,
        resource: DimcatResource[D],
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: Optional[bool] = None,
    ) -> Self:
        """Create a DimcatResource from an existing DimcatResource, specifying its name and, optionally, at what path
        it is to be serialized.

        Args:
            resource:
            resource_name:
            basepath:
            filepath:
            column_schema:
            auto_validate:
                By default, the DimcatResource will not be instantiated if the schema validation fails and the resource
                is re-validated if, for example, the :attr:`column_schema` changes. Set False to prevent validation.
        """
        if not isinstance(resource, DimcatResource):
            raise TypeError(f"Expected a DimcatResource, got {type(resource)!r}.")
        fl_resource = resource.resource.to_copy()
        if resource_name is not None:
            fl_resource.name = resource_name
        if column_schema is not None:
            fl_resource.schema = column_schema
        if basepath is not None:
            resource.basepath = basepath
        if descriptor_filepath is None:
            # if basepath stays the same, also copy the descriptor_filepath (otherwise use default)
            descriptor_filepath = resource.descriptor_filepath
        if filepath is not None:
            fl_resource.path = filepath
        if auto_validate is None:
            auto_validate = resource.auto_validate
        new_object = cls(
            resource=fl_resource,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
        )
        if resource._df is not None:
            new_object._df = resource._df.copy()
        new_object._status = resource._status
        return new_object

    class PickleSchema(Data.Schema):
        resource = mm.fields.Method(
            serialize="get_descriptor_filepath",
            deserialize="raw",
            allow_none=True,
        )
        basepath = mm.fields.String(allow_none=True)

        def get_descriptor_filepath(self, obj: DimcatResource) -> str | dict:
            if obj.is_zipped_resource:
                raise NotImplementedError(
                    f"This {obj.name} is part of a package which currently prevents "
                    f"serializing it as a standalone resource."
                )
            if obj.status < ResourceStatus.DATAFRAME:
                self.logger.debug(
                    f"This {obj.name} is empty and serialized to a dictionary."
                )
                return obj._resource.to_descriptor()
            if obj.status < ResourceStatus.SERIALIZED:
                self.logger.debug(
                    f"This {obj.name} needs to be stored to disk to be expressed as restorable config."
                )
                obj.store_dataframe()
            return obj.get_descriptor_filepath(create_if_necessary=True)

        def raw(self, data):
            return data

        @mm.post_load
        def init_object(self, data, **kwargs):
            if isinstance(data["resource"], str) and "basepath" in data:
                descriptor_path = os.path.join(data["basepath"], data["resource"])
                data["resource"] = descriptor_path
            elif isinstance(data["resource"], dict):
                resource_data = data["resource"]
                _ = resource_data.pop("type", None)
                data["resource"] = fl.Resource(**resource_data)
            return super().init_object(data, **kwargs)

    class Schema(Data.Schema):
        resource = mm.fields.Method(
            serialize="get_resource_descriptor",
            deserialize="raw",
            metadata={"expose": False},
        )
        basepath = mm.fields.String(allow_none=True, metadata={"expose": False})
        descriptor_filepath = mm.fields.String(
            allow_none=True, metadata={"expose": False}
        )
        auto_validate = mm.fields.Boolean(metadata={"expose": False})

        def get_resource_descriptor(self, obj: DimcatResource) -> str | dict:
            return obj._resource.to_descriptor()

        def raw(self, data):
            return data

        @mm.post_load
        def init_object(self, data, **kwargs):
            if "resource" not in data or data["resource"] is None:
                return super().init_object(data, **kwargs)
            if isinstance(data["resource"], str) and "descriptor_filepath" not in data:
                if os.path.isabs(data["resource"]):
                    if "basepath" in data:
                        filepath = make_rel_path(data["resource"], data["basepath"])
                    else:
                        basepath, filepath = os.path.split(data["resource"])
                        data["basepath"] = basepath
                else:
                    filepath = data["resource"]
                data["descriptor_filepath"] = filepath
            if not isinstance(data["resource"], fl.Resource):
                data["resource"] = fl.Resource.from_descriptor(data["resource"])
            return super().init_object(data, **kwargs)

    def __init__(
        self,
        resource: Optional[str, fl.Resource] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """

        Args:
            resource: An existing :obj:`frictionless.Resource` or a file path resolving to a resource descriptor.
            resource_name:
                Name of the resource. Used as filename if the resource is stored to a ZIP file. Defaults to
                :meth:`filename_factory`.
            basepath:
                The absolute path on the local file system, relative to which the resource will be described when
                written to disk. If not specified, it will default to

                * the current working directory if no ``filepath`` is given or the given filepath is relative
                * the ``filepath``'s directory if the filepath is absolute

            filepath:
                The path to the existing or future tabular resource on physical disk.

                * If None it defaults to ``resource_name`` with the extension ``.tsv`` (which also serves as innerpath
                  in case ``filepath`` points to a ZIP file).
                * If it is a relative path, it will be appended to the ``basepath``.
                * If it is an absolute path, the directory will be used as ``basepath``, unless a basepath is specified,
                  in which case filepath must be contained in it, so that it can be expressed relatively to it.

            column_schema:
                If you don't pass a schema or a path or URL to one, frictionless will try to infer it. However,
                this often leads to validation errors.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
        """
        self.logger.debug(
            f"DimcatResource.__init__(resource={resource}, resource_name={resource_name}, basepath={basepath}, "
            f"filepath={filepath}, column_schema={column_schema}, validate={auto_validate})"
        )
        super().__init__()
        self._resource: fl.Resource = make_tsv_resource()
        self._status = ResourceStatus.EMPTY
        self._df: D = None
        self._descriptor_filepath: Optional[str] = descriptor_filepath
        self.auto_validate: bool = auto_validate

        if basepath is not None:
            basepath = ms3.resolve_dir(basepath)

        if resource is not None:
            if isinstance(resource, str):
                descriptor_path = check_file_path(
                    resource, extensions=("resource.json", "resource.yaml")
                )
                fl_resource = fl.Resource(descriptor_path)
                descriptor_dir, descriptor_filepath = os.path.split(descriptor_path)
                self._descriptor_filepath = descriptor_filepath
                if basepath is not None and basepath != descriptor_dir:
                    raise ValueError(
                        f"basepath {basepath!r} does not match the directory of the descriptor {descriptor_path!r}"
                    )
                if not fl_resource.basepath:
                    fl_resource.basepath = descriptor_dir
            elif isinstance(resource, fl.Resource):
                fl_resource = resource
            else:
                raise TypeError(
                    f"Expected a path or a frictionless resource, got {type(resource)!r}"
                )
            self._resource = fl_resource

        if basepath is not None:
            self.basepath = basepath
        elif not self.basepath:
            if filepath is None or not os.path.isabs(filepath):
                self.basepath = get_default_basepath()
                self.logger.info(f"Using default basepath: {self.basepath}")

        if resource_name is not None:
            self.resource_name = resource_name
        if column_schema is not None:
            self.column_schema = column_schema
        if filepath is not None:
            self.filepath = filepath
        elif self.filepath is None:
            self.filepath = self.get_filepath()
        self._update_status()
        if self.auto_validate and self.status == ResourceStatus.DATAFRAME:
            _ = self.validate(raise_exception=True)

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = super().__dir__()
        if self.is_loaded:
            elements.extend(dir(self.df))
        else:
            # if not loaded, expose the field names from the descriptor
            elements.extend(self.field_names)
        return sorted(elements)

    def __getattr__(self, item):
        """Enables using DimcatResource just like the wrapped DataFrame."""
        msg = f"{self.name!r} object ({self._status!r}) has no attribute {item!r}."
        if not self.is_loaded:
            msg += " Try again after loading the dataframe into memory."
            raise AttributeError(msg)
        try:
            return getattr(self.df, item)
        except AttributeError:
            raise AttributeError(msg)

    def __getitem__(self, item):
        if self.is_loaded:
            try:
                return self.df[item]
            except Exception:
                raise KeyError(item)
        elif item in self.field_names:
            raise KeyError(
                f"Column {item!r} will be available after loading the dataframe into memory."
            )
        raise KeyError(item)

    def __len__(self) -> int:
        return len(self.df.index)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return_str = f"{pformat(self.to_dict(), sort_dicts=False)}"
        return f"ResourceStatus={self.status.name}\n{return_str}"

    @property
    def basepath(self) -> str:
        return self._resource.basepath

    @basepath.setter
    def basepath(self, basepath: str):
        basepath = ms3.resolve_dir(basepath)
        if self.is_frozen:
            if basepath == self.basepath:
                return
            if self.descriptor_exists:
                tied_to = f"its descriptor at {self.get_descriptor_path()!r}"
            else:
                tied_to = f"the data stored at {self.normpath!r}"
            raise RuntimeError(
                f"The basepath of resource {self.name!r} ({self.basepath!r}) cannot be changed to {basepath!r} "
                f"because it's tied to {tied_to}."
            )
        assert os.path.isdir(
            basepath
        ), f"Basepath {basepath!r} is not an existing directory."
        self._resource.basepath = basepath
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def column_schema(self) -> fl.Schema:
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
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def descriptor_exists(self) -> bool:
        return self.get_descriptor_path(only_if_exists=True) is not None

    @property
    def descriptor_filepath(self) -> Optional[str]:
        """The path to the descriptor file on disk, relative to the basepath. If you need to fall back to a default
        value, use :meth:`get_descriptor_filepath` instead."""
        return self._descriptor_filepath

    @descriptor_filepath.setter
    def descriptor_filepath(self, descriptor_filepath: str):
        if self.descriptor_exists:
            raise RuntimeError(
                f"Cannot set descriptor filepath for {self.name!r} because it already exists at "
                f"{self.get_descriptor_path()!r}."
            )
        if os.path.isabs(descriptor_filepath):
            filepath = check_file_path(
                descriptor_filepath,
                extensions=("resource.json", "resource.yaml"),
                must_exist=False,
            )
            if self.basepath is None:
                basepath, rel_path = os.path.dirname(filepath)
                self.basepath = basepath
            else:
                rel_path = make_rel_path(filepath, self.basepath)
            self._descriptor_filepath = rel_path
        else:
            self._descriptor_filepath = descriptor_filepath

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
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def field_names(self) -> List[str]:
        """The names of the fields in the resource's schema."""
        return self.column_schema.field_names

    @property
    def filepath(self) -> str:
        return self._resource.path

    @filepath.setter
    def filepath(self, filepath: str):
        if self.is_frozen:
            raise RuntimeError(
                "Cannot set filepath on a resource whose valid descriptor has been written to disk."
            )
        if os.path.isabs(filepath):
            if self.basepath is None:
                basepath, filepath = os.path.split(filepath)
                self.basepath = basepath
                self.logger.info(
                    f"Filepath split into basepath {self.basepath} and filepath {filepath}"
                )
            else:
                filepath = make_rel_path(filepath, self.basepath)
        else:
            if self.basepath is None:
                self.basepath = get_default_basepath()
                self.logger.info(
                    f"Basepath set to current working directory {self.basepath}"
                )
            filepath = check_rel_path(filepath, self.basepath)
        self._resource.path = filepath

    @property
    def innerpath(self) -> str:
        """The innerpath is the resource_name plus the extension .tsv and is used as filename within a .zip archive."""
        if self.resource_name.endswith(".tsv"):
            return self.resource_name
        return self.resource_name + ".tsv"

    @property
    def is_frozen(self) -> bool:
        """Whether the resource is frozen (i.e. its valid descriptor has been written to disk) or not."""
        return self.is_zipped_resource or self.descriptor_exists

    @property
    def is_loaded(self) -> bool:
        return (
            ResourceStatus.DATAFRAME
            <= self.status
            < ResourceStatus.STANDALONE_NOT_LOADED
        )

    @property
    def is_serialized(self) -> bool:
        """Returns True if the resource is serialized (i.e. its dataframe has been written to disk)."""
        if self.normpath is None:
            return False
        if not os.path.isfile(self.normpath):
            return False
        if not self.is_zipped_resource:
            return True
        with zipfile.ZipFile(self.normpath) as zip_file:
            return self.innerpath in zip_file.namelist()

    @property
    def is_valid(self) -> bool:
        report = self.validate(raise_exception=False, only_if_necessary=True)
        if report is None:
            return True
        return report.valid

    @property
    def is_zipped_resource(self) -> bool:
        """Returns True if the filepath points to a .zip file. This means that the resource is part of a package
        and serializes to a dict instead of a descriptor file.
        """
        if self.filepath is None:
            if self.descriptor_filepath is not None and (
                self.descriptor_filepath.endswith("package.json")
                or self.descriptor_filepath.endswith("package.yaml")
            ):
                return True
            return False
        return self.filepath.endswith(".zip")

    @property
    def normpath(self) -> str:
        """Absolute path to the serialized or future tabular file. Raises if basepath is not set."""
        if self.basepath is None:
            raise RuntimeError(f"DimcatResource {self.name} has no basepath.")
        file_path = self.get_filepath()
        return os.path.normpath(os.path.join(self.basepath, file_path))

    @property
    def resource(self) -> fl.Resource:
        return self._resource

    @property
    def resource_name(self) -> str:
        if not self._resource.name:
            return self.filename_factory()
        return self._resource.name

    @resource_name.setter
    def resource_name(self, name: str):
        name_lower = name.lower()
        if not re.match(FRICTIONLESS_NAME_PATTERN, name_lower):
            raise ValueError(
                f"Name must be lowercase and work as filename: {name_lower!r}"
            )
        self._resource.name = name_lower

    @property
    def status(self) -> ResourceStatus:
        if self._status == ResourceStatus.EMPTY and self._resource.schema.fields:
            self._status = ResourceStatus.SCHEMA
        return self._status

    def copy(self) -> Self:
        """Returns a copy of the resource."""
        return self.from_resource(self)

    def get_basepath(self) -> str:
        """Get the basepath of the resource. If not specified, the default basepath is returned."""
        if not self.basepath:
            return get_default_basepath()
        return self.basepath

    def _get_current_status(self) -> ResourceStatus:
        if self.is_zipped_resource:
            if self._df is None:
                return ResourceStatus.PACKAGED_NOT_LOADED
            else:
                return ResourceStatus.PACKAGED_LOADED
        if self.is_serialized:
            if self.descriptor_exists:
                if self._df is None:
                    return ResourceStatus.STANDALONE_NOT_LOADED
                else:
                    return ResourceStatus.STANDALONE_LOADED
            else:
                if self._df is None:
                    self.logger.warning(
                        f"The serialized data exists at {self.normpath} but no descriptor was found at "
                        f"{self.get_descriptor_path(only_if_exists=False)}. Consider passing the "
                        f"descriptor_filepath argument upon initialization."
                    )
                    return ResourceStatus.STANDALONE_NOT_LOADED
                else:
                    return ResourceStatus.SERIALIZED
        elif self._df is not None:
            return ResourceStatus.DATAFRAME
        elif self.column_schema.fields:
            return ResourceStatus.SCHEMA
        return ResourceStatus.EMPTY

    @cache
    def get_dataframe(self) -> Union[D]:
        """
        Load the dataframe from disk based on the descriptor's normpath.

        Returns:
            The dataframe or DimcatResource.
        """
        r = self._resource
        if self.normpath is None:
            raise mm.ValidationError(
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
        if self.status == ResourceStatus.STANDALONE_NOT_LOADED:
            self._status = ResourceStatus.STANDALONE_LOADED
        elif self.status == ResourceStatus.PACKAGED_NOT_LOADED:
            self._status = ResourceStatus.PACKAGED_LOADED
        return df

    def get_descriptor_path(self, only_if_exists: bool = True) -> Optional[str]:
        """Returns the full path to the existing or future descriptor file."""
        if self.basepath is None and only_if_exists:
            return
        descriptor_path = os.path.join(
            self.get_basepath(), self.get_descriptor_filepath()
        )
        if only_if_exists and not os.path.isfile(descriptor_path):
            return
        return descriptor_path

    def get_descriptor_filepath(self, create_if_necessary: bool = False) -> str:
        """Like :attr:`descriptor_filepath` but default to :attr:`filepath` or, if None, to :attr:`innerpath`.
        If ``create_if_necessary=True``, the descriptor will be created in :attr:`basedir` if missing.
        """
        if self.descriptor_filepath is None:
            if self.filepath is None:
                descriptor_filepath = replace_ext(self.innerpath, ".resource.json")
            else:
                descriptor_filepath = replace_ext(self.filepath, ".resource.json")
        else:
            descriptor_filepath = self.descriptor_filepath
        descriptor_path = os.path.join(self.get_basepath(), descriptor_filepath)
        if os.path.isfile(descriptor_path):
            self._descriptor_filepath = descriptor_filepath
        elif create_if_necessary:
            self._descriptor_filepath = descriptor_filepath
            _ = self._store_descriptor()
        return descriptor_filepath

    def get_filepath(self) -> str:
        """Returns the relative path to the data (:attr:`filepath`) if specified, :attr:`innerpath` otherwise."""
        if self.filepath is None:
            return self.innerpath
        return self.filepath

    def get_index(self) -> DimcatIndex:
        return DimcatIndex.from_resource(self)

    def load(self, force_reload: bool = False) -> None:
        """Tries to load the data from disk into RAM. If successful, the .is_loaded property will be True.
        If the resource hadn't been loaded before, its .status property will be updated.
        """
        if not self.is_loaded or force_reload:
            _ = self.df

    def store_dataframe(
        self,
        validate: bool = True,
    ) -> None:
        """Stores the dataframe and its descriptor to disk based on the resource's configuration.

        Raises:
            RuntimeError: If the resource is frozen or does not contain a dataframe or if the file exists already.
        """
        if self.is_frozen:
            raise RuntimeError(
                f"This {self.name} was originally read from disk and therefore is not being stored."
            )
        if self.status < ResourceStatus.DATAFRAME:
            raise RuntimeError(f"This {self.name} does not contain a dataframe.")
        if validate and self.status < ResourceStatus.VALIDATED:
            _ = self.validate(raise_exception=NEVER_STORE_UNVALIDATED_DATA)

        full_path = self.normpath
        if os.path.isfile(full_path):
            raise RuntimeError(
                f"File exists already on disk and will not be overwritten: {full_path}"
            )
        ms3.write_tsv(self.df, full_path)
        self._store_descriptor()
        self._status = ResourceStatus.STANDALONE_LOADED

    def to_config(self, pickle=False) -> DimcatConfig:
        return DimcatConfig(self.to_dict(pickle=pickle))

    def to_dict(self, pickle=False) -> dict:
        if pickle:
            return self.pickle_schema.dump(self)
        return self.schema.dump(self)

    def _store_descriptor(self, overwrite=True) -> str:
        """Stores the descriptor to disk based on the resource's configuration and returns its path.
        Does not modify the resource's :attr:`status`.

        Returns:
            The path to the descriptor file on disk.
        """
        if self.descriptor_filepath is None:
            self.descriptor_filepath = self.get_descriptor_filepath(
                create_if_necessary=False
            )
        descriptor_path = self.get_descriptor_path(only_if_exists=False)
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.debug(
                f"Descriptor exists already and will not be overwritten: {descriptor_path}"
            )
            return descriptor_path
        if descriptor_path.endswith(".resource.yaml"):
            self._resource.to_yaml(descriptor_path)
        elif descriptor_path.endswith(".resource.json"):
            self._resource.to_json(descriptor_path)
        else:
            raise ValueError(
                f"Descriptor path must end with .resource.yaml or .resource.json: {descriptor_path}"
            )
        return descriptor_path

    def _update_status(self) -> None:
        self._status = self._get_current_status()

    def validate(
        self,
        raise_exception: bool = False,
        only_if_necessary: bool = False,
    ) -> Optional[fl.Report]:
        if self.status < ResourceStatus.DATAFRAME:
            self.logger.info("Nothing to validate.")
            return
        if only_if_necessary and self.status >= ResourceStatus.VALIDATED:
            self.logger.info("Already validated.")
            return
        if self.is_serialized:
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
            if self.status == ResourceStatus.VALIDATED:
                self._status = ResourceStatus.DATAFRAME
            raise fl.FrictionlessException("\n".join(errors))
        return report


# endregion DimcatResource


@cache
def get_pickle_schema(name, init=True):
    """Caches the intialized schema for each class. Pass init=False to retrieve the schema constructor."""
    dc_class = get_class(name)
    dc_schema = dc_class.PickleSchema
    if init:
        return dc_schema()
    return dc_schema
