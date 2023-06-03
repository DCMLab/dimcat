from __future__ import annotations

import logging
import os
import zipfile
from enum import IntEnum, auto
from functools import cache
from pprint import pformat
from typing import Dict, Generic, Iterable, List, Optional, TypeAlias, TypeVar, Union

import frictionless as fl
import ms3
import pandas as pd
from dimcat.base import Data, DimcatConfig, get_class, get_setting
from dimcat.utils import check_file_path, check_name, get_default_basepath, replace_ext
from marshmallow import fields, post_load
from typing_extensions import Self

from .utils import (
    align_with_grouping,
    check_rel_path,
    ensure_level_named_piece,
    infer_schema_from_df,
    load_fl_resource,
    load_index_from_fl_resource,
    make_boolean_mask_from_set_of_tuples,
    make_rel_path,
    make_tsv_resource,
)

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


# region DimcatIndex


class IndexField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.to_list()


class DimcatIndex(Generic[IX], Data):
    """A wrapper around a :obj:`pandas.MultiIndex` that provides additional functionality such as keeping track of
    index levels and default groupings.

    A MultiIndex essentially is a Sequence of tuples where each tuple identifies dataframe row and includes one value
    per index level. Each index level has a name and can be seen as in individual :obj:`pandas.Index`. One important
    type of DimcatIndex is the PieceIndex which is a unique MultiIndex (that is, each tuple is unique) and where the
    last (i.e. right-most) level is named `piece`.

    NB: If you want to use the index in a dataframe constructor, use the actual, wrapped index object as in
    `pd.DataFrame(index=dc_index.index)`.
    """

    class Schema(Data.Schema):
        index = IndexField(required=True)
        names = fields.List(fields.Str(), required=True)

        @post_load
        def init_object(self, data, **kwargs) -> pd.MultiIndex:
            return pd.MultiIndex.from_tuples(data["index"], names=data["names"])

    @classmethod
    def from_dataframe(cls, df: SomeDataframe) -> Self:
        """Create a DimcatIndex from a dataframe."""
        return cls.from_index(df.index)

    @classmethod
    def from_index(cls, index: SomeIndex) -> Self:
        """Create a DimcatIndex from a dataframe index."""
        return cls(index)

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
            self._index = pd.MultiIndex.from_tuples([], names=["dimcat_index"])
        elif isinstance(index, pd.Index):
            if None in index.names:
                raise ValueError("Index cannot have a None name: {index.names}.")
            for name in index.names:
                check_name(name)
            self._index = index.copy()
        else:
            raise TypeError(
                f"Expected None or pandas.(Multi)Index, got {type(index)!r}."
            )

    def __contains__(self, item):
        if isinstance(item, tuple):
            return item in set(self._index)
        if isinstance(item, Iterable):
            return set(item).issubset(set(self._index))
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, Iterable):
            return set(self) == set(other)
        return False

    def __getattr__(self, item):
        """Enables using DimcatIndex just like the wrapped Index object."""
        return getattr(self._index, item)

    def __getitem__(self, item):
        """Enables using DimcatIndex just like the wrapped Index object."""
        result = self._index[item]
        if isinstance(result, pd.Index):
            return self.__class__(result)
        return result

    def __hash__(self):
        return hash(set(self._index))

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return str(self._index)

    @property
    def index(self) -> IX:
        return self._index

    @property
    def names(self) -> List[str]:
        return list(self._index.names)

    @property
    def piece_level_position(self) -> Optional[int]:
        """The position of the `piece` level in the index, or None if the index has no `piece` level."""
        return self.names.index("piece") if "piece" in self.names else None

    def copy(self) -> Self:
        return self.__class__(self._index.copy())

    def to_resource(self, **kwargs) -> DimcatResource:
        """Create a DimcatResource from this index."""
        return DimcatResource.from_index(self, **kwargs)


class PieceIndex(DimcatIndex[IX]):
    """A unique DimcatIndex where the last (i.e. right-most) level is named `piece`."""

    @classmethod
    def from_index(
        cls,
        index: DimcatIndex[IX] | IX,
        recognized_piece_columns: Optional[Iterable[str]] = None,
        max_levels: int = 2,
    ) -> Self:
        """Create a PieceIndex from another index."""
        if isinstance(index, DimcatIndex):
            index = index.index
        if len(index) == 0:
            return cls()
        index, piece_level_position = ensure_level_named_piece(
            index, recognized_piece_columns
        )
        level_names = index.names
        right_boundary = piece_level_position + 1
        drop_levels = level_names[right_boundary:]
        if piece_level_position >= max_levels:
            drop_levels = level_names[: right_boundary - max_levels] + drop_levels
        if len(drop_levels) > 0:
            index = index.droplevel(drop_levels)
        return cls(index)

    @classmethod
    def from_resource(
        cls,
        resource: DimcatResource | fl.Resource,
        index_col: Optional[int | str | List[int | str]] = None,
        recognized_piece_columns: Optional[Iterable[str]] = None,
        max_levels: int = 2,
    ) -> Self:
        """Create a PieceIndex from a frictionless Resource."""
        index = DimcatIndex.from_resource(
            resource,
            index_col=index_col,
        )
        return cls.from_index(
            index,
            recognized_piece_columns=recognized_piece_columns,
            max_levels=max_levels,
        )

    def __init__(self, index: Optional[IX] = None):
        if index is None:
            index = pd.MultiIndex.from_tuples([], name=("corpus", "piece"))
        else:
            index = index.drop_duplicates()
            assert (
                index.names[-1] == "piece"
            ), f"Expected last level to be named 'piece', got {index.names[-1]!r}."
        super().__init__(index)


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
    def from_index(
        cls,
        index: DimcatIndex | SomeIndex,
        resource_name: str,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> Self:
        if isinstance(index, DimcatIndex):
            index = index.index
        dataframe = pd.DataFrame(index=index)
        return cls.from_dataframe(
            df=dataframe,
            resource_name=resource_name,
            basepath=basepath,
            filepath=filepath,
            column_schema=column_schema,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
        )

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
        resource = fields.Method(
            serialize="get_descriptor_filepath",
            deserialize="raw",
            allow_none=True,
        )
        basepath = fields.String(allow_none=True)

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

        @post_load
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
        resource = fields.Method(
            serialize="get_resource_descriptor",
            deserialize="raw",
            metadata={"expose": False},
        )
        basepath = fields.String(allow_none=True, metadata={"expose": False})
        descriptor_filepath = fields.String(allow_none=True, metadata={"expose": False})
        auto_validate = fields.Boolean(metadata={"expose": False})

        def get_resource_descriptor(self, obj: DimcatResource) -> str | dict:
            return obj._resource.to_descriptor()

        def raw(self, data):
            return data

        @post_load
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
        return os.path.isfile(self.get_descriptor_path())

    @property
    def descriptor_filepath(self) -> Optional[str]:
        """The path to the descriptor file on disk, relative to the basepath. If you need to fall back to a default
        value, use :meth:`get_descriptor_filepath` instead."""
        return self._descriptor_filepath

    @descriptor_filepath.setter
    def descriptor_filepath(self, descriptor_filepath: str):
        if self.descriptor_exists:
            if (
                descriptor_filepath == self._descriptor_filepath
                or descriptor_filepath == self.get_descriptor_path()
            ):
                self.logger.info(
                    f"Descriptor filepath for {self.name!r} was already set to {descriptor_filepath!r}."
                )
            else:
                raise RuntimeError(
                    f"Cannot set descriptor_filepath for {self.name!r} to {descriptor_filepath} because it already "
                    f"set to the existing one at {self.get_descriptor_path()!r}."
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
    def is_empty(self) -> bool:
        return self.status < ResourceStatus.DATAFRAME

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
        if self.basepath is None:
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
            raise RuntimeError(f"DimcatResource {self.resource_name} has no basepath.")
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
    def resource_name(self, resource_name: str):
        check_name(resource_name)
        self._resource.name = resource_name

    @property
    def status(self) -> ResourceStatus:
        if self._status == ResourceStatus.EMPTY and self._resource.schema.fields:
            self._status = ResourceStatus.SCHEMA
        return self._status

    def align_with_grouping(
        self,
        grouping: DimcatIndex | pd.MultiIndex,
        sort_index=True,
    ) -> pd.DataFrame:
        """Aligns the resource with a grouping index. In the typical case, the grouping index will come with the levels
        ["<grouping_name>", "corpus", "piece"] and the result will be aligned such that every group contains the
        resource's sub-dataframes for the included pieces.
        """
        if isinstance(grouping, DimcatIndex):
            grouping = grouping.index
        if self.is_empty:
            self.logger.warning(f"Resource {self.name} is empty.")
            return pd.DataFrame(index=grouping)
        return align_with_grouping(self.df, grouping, sort_index=sort_index)

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
        dataframe = load_fl_resource(self._resource)
        if self.status == ResourceStatus.STANDALONE_NOT_LOADED:
            self._status = ResourceStatus.STANDALONE_LOADED
        elif self.status == ResourceStatus.PACKAGED_NOT_LOADED:
            self._status = ResourceStatus.PACKAGED_LOADED
        return dataframe

    def get_descriptor_path(self, create_if_necessary: bool = False) -> Optional[str]:
        """Returns the full path to the existing or future descriptor file."""
        descriptor_path = os.path.join(
            self.get_basepath(), self.get_descriptor_filepath()
        )
        if not os.path.isfile(descriptor_path) and create_if_necessary:
            self._store_descriptor(descriptor_path)
        return descriptor_path

    def get_descriptor_filepath(self) -> str:
        """Like :attr:`descriptor_filepath` but returning a default value if None."""
        if self.descriptor_filepath is not None:
            return self.descriptor_filepath
        if self.filepath is None:
            descriptor_filepath = replace_ext(self.innerpath, ".resource.json")
        else:
            descriptor_filepath = replace_ext(self.filepath, ".resource.json")
        return descriptor_filepath

    def get_filepath(self) -> str:
        """Returns the relative path to the data (:attr:`filepath`) if specified, :attr:`innerpath` otherwise."""
        if self.filepath is None:
            return self.innerpath
        return self.filepath

    def get_index(self) -> DimcatIndex:
        """Returns the index of the resource based on the ``primaryKey`` of the :obj:`frictionless.Schema`."""
        return DimcatIndex.from_resource(self)

    def get_path_dict(self) -> Dict[str, str]:
        """Returns a dictionary with the paths to the resource's data and descriptor."""
        return dict(
            basepath=self.basepath,
            filepath=self.filepath,
            descriptor_filepath=self.descriptor_filepath,
            normpath=self.normpath,
            descriptor_path=self.get_descriptor_path(),
        )

    def get_piece_index(self, max_levels: int = 2) -> PieceIndex:
        """Returns the :class:`PieceIndex` of the resource based on :attr:`get_index`. That is,
        an index of which the right-most level is unique and called `piece` and up to ``max_levels``
        additional index levels to its right.

        Args:
            max_levels: By default, the number of levels is limited to the default 2, ('corpus', 'piece').

        Returns:
            An index of the pieces described by the resource.
        """
        return PieceIndex.from_resource(self, max_levels=max_levels)

    def load(self, force_reload: bool = False) -> None:
        """Tries to load the data from disk into RAM. If successful, the .is_loaded property will be True.
        If the resource hadn't been loaded before, its .status property will be updated.
        """
        if not self.is_loaded or force_reload:
            _ = self.df

    def subselect(
        self,
        tuples: DimcatIndex | Iterable[tuple],
        levels: Optional[int | str | List[int | str]] = None,
    ) -> pd.DataFrame:
        """Returns a copy of a subselection of the dataframe based on the union of its index tuples (or subtuples)
        and the given tuples."""
        if self.is_empty:
            self.logger.warning("Resource is empty.")
            return self.copy()
        tuple_set = set(tuples)
        random_tuple = next(iter(tuple_set))
        if not isinstance(random_tuple, tuple):
            raise TypeError(
                f"Pass an iterable of tuples. A randomly selected element had type {type(random_tuple)!r}."
            )
        mask = make_boolean_mask_from_set_of_tuples(self.df.index, tuple_set, levels)
        return self.df[mask].copy()

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
            _ = self.validate(raise_exception=True)

        full_path = self.normpath
        if os.path.isfile(full_path):
            raise RuntimeError(
                f"File exists already on disk and will not be overwritten: {full_path}"
            )
        ms3.write_tsv(self.df, full_path)
        self._store_descriptor()
        self._status = ResourceStatus.STANDALONE_LOADED

    def _store_descriptor(
        self, descriptor_path: Optional[str] = None, overwrite=True
    ) -> str:
        """Stores the descriptor to disk based on the resource's configuration and returns its path.
        Does not modify the resource's :attr:`status`.

        Returns:
            The path to the descriptor file on disk.
        """
        if descriptor_path is None:
            descriptor_path = self.get_descriptor_path()
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.info(
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
        self.descriptor_filepath = descriptor_path
        return descriptor_path

    def to_config(self, pickle=False) -> DimcatConfig:
        return DimcatConfig(self.to_dict(pickle=pickle))

    def to_dict(self, pickle=False) -> dict:
        if pickle:
            return self.pickle_schema.dump(self)
        return self.schema.dump(self)

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
        else:
            errors = [err.message for task in report.tasks for err in task.errors]
            if self.status == ResourceStatus.VALIDATED:
                self._status = ResourceStatus.DATAFRAME
            if get_setting("never_store_unvalidated_data") and raise_exception:
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
