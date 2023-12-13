from __future__ import annotations

import logging
import os
import warnings
from functools import cache
from numbers import Number
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    Union,
    overload,
)

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import (
    DO,
    DimcatConfig,
    FriendlyEnum,
    get_class,
    get_setting,
    resolve_object_specs,
)
from dimcat.data.base import Data
from dimcat.data.resources.base import (
    IX,
    D,
    F,
    FeatureName,
    Resource,
    ResourceStatus,
    Rs,
    SomeDataframe,
    SomeIndex,
)
from dimcat.data.resources.utils import (
    align_with_grouping,
    apply_slice_intervals_to_resource_df,
    ensure_level_named_piece,
    feature_specs2config,
    get_time_spans_from_resource_df,
    infer_schema_from_df,
    load_fl_resource,
    load_index_from_fl_resource,
    make_boolean_mask_from_set_of_tuples,
    make_index_from_grouping_dict,
    make_tsv_resource,
    resolve_levels_argument,
)
from dimcat.dc_exceptions import (
    BasePathNotDefinedError,
    DataframeIncompatibleWithColumnSchemaError,
    FeatureUnavailableError,
    FilePathNotDefinedError,
    PotentiallyUnrelatedDescriptorError,
    ResourceIsFrozenError,
)
from dimcat.dc_warnings import PotentiallyUnrelatedDescriptorUserWarning
from dimcat.utils import check_name
from frictionless import FrictionlessException
from plotly import graph_objs as go
from typing_extensions import Literal, Self

if TYPE_CHECKING:
    from dimcat.data.resources.results import Result
    from dimcat.steps.base import StepSpecs

# region DimcatResource

resource_status_logger = logging.getLogger("dimcat.data.resources.ResourceStatus")
levelvalue_: TypeAlias = Union[str, Number, bool]


class UnitOfAnalysis(FriendlyEnum):
    SLICE = "SLICE"
    PIECE = "PIECE"
    GROUP = "GROUP"


class DimcatResource(Resource, Generic[D]):
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
      Its directory is used as ``basepath``. ``descriptor_path`` is expected to end in "resource.[json|yaml]".
    - ``DimcatResource.from_dataframe(df=df, resource_name, basepath)``: Creates a new DimcatResource from a dataframe.
      If ``basepath`` is not specified, the current working directory is used if the resource is to be serialized.
    - ``DimcatResource.from_resource(resource=DimcatResource)``: Creates a DimcatResource from an existing one
      by copying the fields it specifies.

    Resource in a package (common case)
    -----------------------------------

    A DimcatResource knows that it is part of a package if its ``filepath`` ends on ``.zip``. In that case, the
    DimcatPackage will take care of the serialization and not store an individual resource descriptor.
    """

    # region column name class variables
    _auxiliary_column_names: ClassVar[Optional[List[str]]] = None
    """Names of columns that specify additional properties of the objects (each row is one object) but which are not
    required. E.g., the color of an annotation label."""
    _convenience_column_names: ClassVar[Optional[List[str]]] = None
    """Names of columns containing other representations of the objects (each row is one object) which can be computed
    from the feature columns in case they are missing."""
    _default_value_column: Optional[ClassVar[str]] = None
    """Name of the column containing representative values for this resource. For example, they could be chosen as
    values to be tallied up and displayed along the x-axis of a bar plot. If the :attr:`value_column` has not been set,
    it returns this column name. For :obj:`Features <Feature>`, the value may default to the last element of
    :attr:`_feature_columns`, if defined.
    """
    _default_formatted_column: Optional[ClassVar[str]] = None
    """A secondary value column that represents the :attr:`_default_value_column` in a different format. This is
    often one of the :attr:`_convenience_column_names`."""
    _feature_column_names: ClassVar[Optional[List[str]]] = None
    """Name(s) of the column(s) which are required to fully define an individual object (each row is an object). When
    creating the resource, any row containing a missing value in one of the feature columns is dropped."""
    # endregion column name class variables
    # region associated object types
    _default_analyzer: ClassVar[StepSpecs] = "Proportions"
    """Name of the Analyzer that is used by default for plotting the resource. Needs to return a :obj:`Result`."""
    _extractable_features: ClassVar[Optional[Tuple[FeatureName, ...]]] = None
    """Tuple of :obj:`FeatureNames <FeatureName>` corresponding to the features that can be extracted from this
    resource. If None, no features can be extracted."""
    # endregion associated object types

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | fl.Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
    ) -> Self:
        """Create a DimcatResource by loading its frictionless descriptor from disk.
        The descriptor's directory is used as ``basepath``. ``descriptor_path`` is expected to end in
        ``.resource.json``.

        Args:
            descriptor: Descriptor corresponding to a frictionless resource descriptor.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        return super().from_descriptor(
            descriptor=descriptor,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            **kwargs,
        )

    @classmethod
    def from_descriptor_path(
        cls,
        descriptor_path: str,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a frictionless descriptor file on disk.

        Args:
            descriptor_path: Absolute path where the JSON/YAML descriptor is located.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).

        """
        return super().from_descriptor_path(
            descriptor_path=descriptor_path,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: D,
        resource_name: str,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
    ) -> Self:
        """Create a DimcatResource from a dataframe, specifying its name and, optionally, at what path it is to be
        serialized.

        Args:
            df: Dataframe to create the resource from.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        new_object = cls(
            basepath=basepath,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            **kwargs,
        )
        if resource_name is not None:
            new_object.resource_name = resource_name
        new_object.set_dataframe(df)
        return new_object

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        basepath: Optional[str] = None,
        **kwargs: Optional[bool],
    ) -> Self:
        """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.

        Args:
            filepath: Path pointing to a resource descriptor or a simple path resource.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
            basepath:
                Basepath to use for the resource. If None, the folder of the ``filepath`` is used.
        """
        return super().from_filepath(
            filepath=filepath,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_index(
        cls,
        index: DimcatIndex | SomeIndex,
        resource_name: str,
        basepath: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        if isinstance(index, DimcatIndex):
            index = index.index
        dataframe = pd.DataFrame(index=index)
        return cls.from_dataframe(
            df=dataframe,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            basepath=basepath,
        )

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        descriptor_filename: Optional[str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: Optional[bool] = None,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
    ) -> Self:
        """Create a DimcatResource from an existing :obj:`Resource`, specifying its name and,
        optionally, at what path it is to be serialized.

        Args:
            resource: An existing :obj:`frictionless.Resource` or a filepath.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
        new_object = super().from_resource(
            resource=resource,
            descriptor_filename=descriptor_filename,
            resource_name=resource_name,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            **kwargs,
        )
        # copy additional fields
        for attr in ("_df", "_status", "_corpus_name", "_default_groupby"):
            if (
                hasattr(resource, attr)
                and (value := getattr(resource, attr)) is not None
            ):
                setattr(new_object, attr, value)
        return new_object

    @classmethod
    def from_resource_and_dataframe(
        cls,
        resource: Resource,
        df: D,
        descriptor_filename: Optional[str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: Optional[bool] = None,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
    ) -> Self:
        """Create a DimcatResource from an existing :obj:`Resource`, specifying its name and,
        optionally, at what path it is to be serialized.

        Args:
            resource: An existing :obj:`frictionless.Resource` or a filepath.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
        new_object = super().from_resource(
            resource=resource,
            descriptor_filename=descriptor_filename,
            resource_name=resource_name,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            **kwargs,
        )
        if not descriptor_filename and new_object.descriptor_exists:
            new_object.detach_from_descriptor()
        if new_object.resource_exists:
            new_object.detach_from_filepath()
        # copy additional fields
        for attr in ("_corpus_name",):
            if (
                hasattr(resource, attr)
                and (value := getattr(resource, attr)) is not None
            ):
                setattr(new_object, attr, value)
        new_object.set_dataframe(df)
        return new_object

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        **kwargs,
    ) -> Self:
        if not resource_path.endswith(".tsv"):
            fname, fext = os.path.splitext(os.path.basename(resource_path))
            raise NotImplementedError(
                f"{fname}: Don't know how to load {fext} files yet."
                f"Either load the resource yourself and use {cls.name}.from_dataframe() or, if you "
                f"want to get a simple path resource, use Resource.from_resource_path() (not "
                f"DimcatResource)."
            )
        df = ms3.load_tsv(resource_path)
        return cls.from_dataframe(
            df=df,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            **kwargs,
        )

    @classmethod
    def get_default_column_names(
        cls, include_context_columns: bool = True
    ) -> List[str]:
        """Returns the default column names for a DimcatResource."""
        column_names = []
        if include_context_columns:
            column_names.extend(get_setting("context_columns"))
        if cls._auxiliary_column_names:
            column_names.extend(cls._auxiliary_column_names)
        if cls._convenience_column_names:
            column_names.extend(cls._convenience_column_names)
        if cls._feature_column_names:
            column_names.extend(cls._feature_column_names)
        return column_names

    class Schema(Resource.Schema):
        auto_validate = mm.fields.Boolean(
            metadata=dict(
                expose=False,
                description="By default, the DimcatResource will not be validated upon instantiation or change (but "
                "always before writing to disk). Set True to raise an exception during creation or "
                "modification of the resource, e.g. replacing the :attr:`column_schema`.",
            )
        )
        default_groupby = mm.fields.List(
            mm.fields.String(),
            allow_none=True,
            metadata=dict(
                expose=False,
                description="Pass a list of column names or index levels to groupby something else than the default "
                "(by piece).",
            ),
        )

        # @mm.post_load
        # def init_object(self, data, **kwargs):
        #     if "resource" not in data or data["resource"] is None:
        #         return super().init_object(data, **kwargs)
        #     if isinstance(data["resource"], str) and "descriptor_filename" not in data:
        #         if os.path.isabs(data["resource"]):
        #             if "basepath" in data:
        #                 filepath = make_rel_path(data["resource"], data["basepath"])
        #             else:
        #                 basepath, filepath = os.path.split(data["resource"])
        #                 data["basepath"] = basepath
        #         else:
        #             filepath = data["resource"]
        #         data["descriptor_filename"] = filepath
        #     if not isinstance(data["resource"], fl.Resource):
        #         data["resource"] = fl.Resource.from_descriptor(data["resource"])
        #     return super().init_object(data, **kwargs)

    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        self.logger.debug(
            f"""
DimcatResource.__init__(
    resource={type(resource)},
    descriptor_filename={descriptor_filename!r},
    basepath={basepath!r},
    auto_validate={auto_validate!r},
    default_groupby={default_groupby!r},
)"""
        )
        self._df: D = None
        self.auto_validate = True if auto_validate else False  # catches None
        self._default_groupby: List[str] = []
        self._value_column: Optional[str] = None
        self._formatted_column: Optional[str] = None
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
        )
        if default_groupby is not None:
            self.default_groupby = default_groupby

        if self.auto_validate and self.status == ResourceStatus.DATAFRAME:
            _ = self.validate(raise_exception=True)

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = list(super().__dir__())
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
            raise AttributeError(
                f"AttributeError: {self.name!r} object has no attribute {item!r}"
            )

    def __getitem__(self, item):
        if self.is_loaded:
            try:
                return self.df[item]
            except Exception as e:
                raise KeyError(item) from e
        elif item in self.field_names:
            raise KeyError(
                f"Column {item!r} will be available after loading the dataframe into memory."
            )
        raise KeyError(item)

    def __len__(self) -> int:
        return len(self.df.index)

    @property
    def column_schema(self) -> fl.Schema:
        return self._resource.schema

    @column_schema.setter
    def column_schema(self, new_schema: fl.Schema):
        if self.is_frozen:
            raise ResourceIsFrozenError(
                message="Cannot set schema on a resource whose valid descriptor has been written to disk."
            )
        self._resource.schema = new_schema
        status_before = self.status
        if self.status < ResourceStatus.SCHEMA_ONLY:
            self._status = ResourceStatus.SCHEMA_ONLY
        elif self.status >= ResourceStatus.VALIDATED:
            self._status = ResourceStatus.DATAFRAME
        if self.status != status_before:
            resource_status_logger.debug(
                f"After setting the column schema of {self.resource_name!r}, the status has been "
                f"changed from {status_before!r} to {self._status!r}."
            )
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def default_groupby(self) -> List[str]:
        return list(self._default_groupby)

    @default_groupby.setter
    def default_groupby(self, default_groupby: str | List[str]) -> None:
        if default_groupby is None:
            raise ValueError("default_groupby cannot be None")
        if isinstance(default_groupby, str):
            default_groupby = [default_groupby]
        else:
            default_groupby = list(default_groupby)
        if self.is_loaded:
            available_levels = self.get_level_names()
            missing = [
                level for level in default_groupby if level not in available_levels
            ]
            if missing:
                raise ValueError(
                    f"Invalid default_groupby: {missing!r} are not valid levels. "
                    f"Available levels are: {available_levels!r}"
                )
        self._default_groupby = default_groupby

    @property
    def df(self) -> D:
        if self._df is not None:
            resource_df = self._df
        elif self.is_serialized:
            resource_df = self.get_dataframe()
            self._set_dataframe(resource_df)
        else:
            RuntimeError(f"No dataframe accessible for this {self.name}:\n{self}")
        return resource_df

    @df.setter
    def df(self, df: D) -> None:
        self.set_dataframe(df)

    @property
    def extractable_features(self) -> Tuple[FeatureName, ...]:
        if self._extractable_features is None:
            return tuple()
        return tuple(self._extractable_features)

    @property
    def field_names(self) -> List[str]:
        """The names of the fields in the resource's schema."""
        return self.column_schema.field_names

    @property
    def formatted_column(self) -> Optional[str]:
        """A secondary value column that represents the :attr:`value_column` in a different format. If it hasn't been
        set, it defaults to :attr:`_default_formatted_column`, falling back to :attr:`value_column`.
        """
        if self._formatted_column is not None:
            return self._formatted_column
        if self._default_formatted_column is not None:
            return self._default_formatted_column
        return

    @property
    def has_distinct_formatted_column(self) -> bool:
        """Returns False if no formatted_column is specified or it is identical with :attr:`value_column`."""
        return self.formatted_column and self.formatted_column != self.value_column

    @property
    def innerpath(self) -> str:
        """The innerpath is the resource_name plus the extension .tsv and is used as filename within a .zip archive."""
        if self.resource_name.endswith(".tsv"):
            return self.resource_name
        return self.resource_name + ".tsv"

    @property
    def is_empty(self) -> bool:
        """Whether this resource holds data available or not (yet)."""
        return self.status < ResourceStatus.DATAFRAME

    @property
    def is_loaded(self) -> bool:
        return (
            self._df is not None
            or ResourceStatus.SCHEMA_ONLY
            < self.status
            < ResourceStatus.STANDALONE_NOT_LOADED
        )

    @property
    def is_valid(self) -> bool:
        """Returns the result of a previous validation or, if the resource has not been validated
        before, do it now. Importantly, this property assumes serialized resoures to be valid. If
        you want to actively validate the resource, use :meth:`validate` instead."""
        if self.is_serialized:
            return True
        return super().is_valid

    @property
    def value_column(self) -> Optional[str]:
        """Name of the column containing representative values for this resource. If not set, it defaults to
        :attr:`_default_value_column`, falling back to the last element of :attr:`_feature_columns`, if defined.
        """
        if self._value_column is not None:
            return self._value_column
        if self._default_value_column is not None:
            return self._default_value_column
        if self._feature_column_names is not None:
            return self._feature_column_names[-1]
        return

    def align_with_grouping(
        self,
        grouping: DimcatIndex | pd.MultiIndex,
        sort_index=True,
    ) -> pd.DataFrame:
        """Aligns the resource with a grouping index. In the typical case, the grouping index will come with the levels
        ["<grouping_name>", "corpus", "piece"] and the result will be aligned such that every group contains the
        resource's sub-dataframes for the included pieces.
        """
        if self.is_empty:
            self.logger.warning(f"Resource {self.name} is empty.")
            return pd.DataFrame(index=grouping)
        return align_with_grouping(self.df, grouping, sort_index=sort_index)

    def apply_slice_intervals(
        self,
        slice_intervals: SliceIntervals | pd.MultiIndex,
    ) -> pd.DataFrame:
        """"""
        if isinstance(slice_intervals, DimcatIndex):
            slice_intervals = slice_intervals.index
        if self.is_empty:
            self.logger.warning(f"Resource {self.name} is empty.")
            return pd.DataFrame(index=slice_intervals)
        return apply_slice_intervals_to_resource_df(
            df=self.df, slice_intervals=slice_intervals, logger=self.logger
        )

    @overload
    def apply_step(self, step: StepSpecs | List | Tuple) -> DO:
        ...

    @overload
    def apply_step(self, *step: StepSpecs) -> DO:
        ...

    def apply_step(self, *step: StepSpecs) -> DO:
        """Applies one or several pipeline steps to this resource. For backward compatibility, when only a single
        argument is passed, the method accepts it to be a list or tuple of step specs, too.
        """
        if len(step) == 1:
            single_step = step[0]
            if isinstance(single_step, (list, tuple)):
                return self.apply_step(*single_step)
            step_obj = resolve_object_specs(single_step, "PipelineStep")
            return step_obj.process_resource(self)
        Constructor = get_class("Pipeline")
        pipeline = Constructor(steps=step)
        return pipeline.process_resource(self)

    def _check_feature_config(self, feature_config: DimcatConfig) -> None:
        """
        Check whether a feature that is compatible with the given configuration can be extracted from this resource.
        """
        feature_name = feature_config.options_dtype
        if feature_name not in self.extractable_features:
            raise FeatureUnavailableError(feature_name, self.resource_name)

    def _drop_rows_with_missing_values(
        self,
        df: D,
        column_names: Optional[List[str]] = None,
        how: Literal["any", "all"] = "any",
    ) -> D:
        """Drop rows with missing values in the specified columns. If nothing is to be dropped, the identical
        dataframe is returned, not a copy. Falls back to the feature columns if no columns are specified or,
        if no feature columns are defined, nothing is dropped.
        """
        if not column_names:
            if self._feature_column_names:
                column_names = self._feature_column_names
            else:
                self.logger.debug(
                    f"No feature columns defined for {self.resource_name}. Returning as is."
                )
                return df
        if how == "any":
            drop_mask = df[column_names].isna().any(axis=1)
        elif how == "all":
            drop_mask = df[column_names].isna().all(axis=1)
        else:
            raise ValueError(
                f"Invalid value for how: {how!r}. Expected either 'how' or 'all'."
            )
        if drop_mask.all():
            raise RuntimeError(
                f"The {self.name} {self.resource_name!r} contains no fully defined objects based on the "
                f"columns {column_names}."
            )
        n_dropped = drop_mask.sum()
        if n_dropped:
            df = df.dropna(subset=column_names)
            self.logger.info(
                f"Dropped {n_dropped} rows from {self.resource_name} that pertaine to segments following the last "
                f"cadence label in the piece."
            )
        return df

    def extract_feature(
        self,
        feature: FeatureSpecs,
        new_name: Optional[str] = None,
    ) -> F:
        feature_config = feature_specs2config(feature)
        self._check_feature_config(feature_config)
        return self._extract_feature(feature_config=feature_config, new_name=new_name)

    def _extract_feature(
        self,
        feature_config: DimcatConfig,
        new_name: Optional[str] = None,
    ) -> F:
        """The internal part of the feature extraction that subclasses can override to perform certain transformations
        necessary for creating the Feature.
        """
        feature_name = feature_config.options_dtype
        Constructor = get_class(feature_name)
        if new_name is None:
            new_name = f"{self.resource_name}.{feature_name.lower()}"
        feature_df = self._prepare_feature_df(feature_config)
        len_before = len(feature_df)
        feature_df = self._transform_feature_df(feature_df, feature_config)
        init_args = dict(
            resource_name=new_name,
            descriptor_filename=None,
            basepath=None,
            auto_validate=self.auto_validate,
            default_groupby=self.default_groupby,
        )
        init_args.update(feature_config.init_args)
        feature = Constructor.from_dataframe(df=feature_df, **init_args)
        len_after = len(feature.df)
        self.logger.debug(
            f"Create {Constructor.name} with {len_after} rows from {self.name} {self.resource_name!r} of length "
            f"{len_before}."
        )
        return feature

    def filter_index_level(
        self,
        keep_values: levelvalue_ | Iterable[levelvalue_] = None,
        drop_values: levelvalue_ | Iterable[levelvalue_] = None,
        level: int | str = 0,
        drop_level: Optional[bool] = None,
    ) -> Self:
        """Returns a copy of the resource with only those rows where the given level has desired values.

        Args:
            keep_values:
                One or several values to keep (dropping the rest). If a value is specified both for keeping and
                dropping, it is dropped.
            drop_values: One or several values to drop.
            level: Which index level to filter on.
            drop_level:
                Boolean specifies whether to keep the filtered level or to drop it. The default (None) corresponds
                to automatic behaviour, where the level is dropped if only one value remains, otherwise kept.

        Returns:
            A copy of the resource with only those rows where the given level has desired values.
        """
        if not isinstance(level, (int, str)):
            raise TypeError(
                f"Level must be an int position or name string, got {type(level)!r}."
            )
        idx = self.get_index()
        drop_this, keep_values = idx.get_level_values_to_drop(
            drop_values, keep_values, level
        )
        do_level_drop = drop_level or (drop_level is None and len(keep_values) < 2)
        if not (drop_this or do_level_drop):
            self.logger.info(
                f"Nothing to filter based on keep_values={keep_values} and drop_values={drop_values}."
            )
            return self.copy()
        if drop_this:
            new_df = self.df.drop(drop_this, level=level)
        else:
            new_df = self.df
        if do_level_drop:
            new_df = new_df.droplevel(level)
        new_resource = self.__class__.from_resource_and_dataframe(
            resource=self, df=new_df
        )
        if do_level_drop and level in new_resource.default_groupby:
            new_resource._default_groupby.remove(level)
        return new_resource

    def _format_dataframe(self, df: D) -> D:
        """Format the dataframe before it is set for this resource. The method is called by :meth:`_set_dataframe`
        and typically adds convenience columns. Assumes that the dataframe can be mutated safely, i.e. that it is a
        copy.

        Most features have a line such as

        .. code-block:: python

            df = df._drop_rows_with_missing_values(df, column_names=self._feature_column_names)

        to keep only fully defined objects. The index is not reset to retain
        traceability to the original facet. In some cases, the durations need to adjusted when dropping rows. For
        example, 'adjacency groups', i.e., subsequent identical values, can be merged using the pattern

        .. code-block:: python

            group_keys, _ = make_adjacency_groups(<feature column(s)>, groupby=<groupby_levels>)
            feature_df = condense_dataframe_by_groups(df, group_keys)

        """
        return df

    def _get_current_status(self) -> ResourceStatus:
        if self.is_packaged:
            if self.is_loaded:
                return ResourceStatus.PACKAGED_LOADED
            else:
                return ResourceStatus.PACKAGED_NOT_LOADED
        match (self.is_serialized, self.descriptor_exists, self.is_loaded):
            case (True, True, True):
                return ResourceStatus.STANDALONE_LOADED
            case (True, True, False):
                return ResourceStatus.STANDALONE_NOT_LOADED
            case (True, False, True):
                return ResourceStatus.SERIALIZED
            case (True, False, False):
                # warnings.warn(
                #     f"The serialized data exists at {self.normpath!r} but no descriptor was found at "
                #     f"{self.get_descriptor_path()!r}. You can create one using .store_descriptor(), set the "
                #     f"descriptor_filename pointing to one (should be done upon instantiation), or, if this is
                #     supposed to be a PathResource only, it should not be instantiated as DimcatResource at all.",
                #     RuntimeWarning,
                # )
                return ResourceStatus.PATH_ONLY
            case (False, _, True):
                if self.descriptor_exists:
                    if not self.filepath:
                        raise RuntimeError(
                            f"The resource points to an existing descriptor at {self.get_descriptor_path()!r} but "
                            f"no filepath has been set. This should not have happened. Please consider filing an issue."
                        )
                    warnings.warn(
                        f"The resource is loaded and the there exists a descriptor at {self.get_descriptor_path()!r}, "
                        f"but the normpath {self.normpath} does not exist. This could signify a mismatch between the "
                        f"loaded dataframe and the data described by the descriptor which could result in data loss if "
                        f"the dataframe is serialized to disk, overwriting the descriptor that was actually describing "
                        f"something else.",
                        PotentiallyUnrelatedDescriptorUserWarning,
                    )
                if self._is_valid:  # using the property could trigger validation
                    return ResourceStatus.VALIDATED
                return ResourceStatus.DATAFRAME
            case _:
                if self.basepath and self.descriptor_exists:
                    warnings.warn(
                        f"The resource points to an existing descriptor at {self.get_descriptor_path()!r} but it "
                        f"hasn't been loaded. Please consider passing discriptor_filename="
                        f"{self.get_descriptor_filename()} when instantiating or using {self.name}"
                        f".from_descriptor_path(). If this is what you did, this warning likely stems from a bug, "
                        f"please consider filing an issue in this case.",
                        PotentiallyUnrelatedDescriptorUserWarning,
                    )
                if self.column_schema.fields:
                    return ResourceStatus.SCHEMA_ONLY
                return ResourceStatus.EMPTY

    @cache
    def get_dataframe(
        self,
        index_col: Optional[int | str | Tuple[int | str]] = None,
        usecols: Optional[int | str | Tuple[int | str]] = None,
    ) -> D:
        """
        Load the dataframe from disk based on the descriptor's normpath. This does not change the resource's status.

        Args:
            index_col:
                Can be used to override the primary_key(s) specified in the resource's schema.
                Value(s) can be column name(s) or column position(s), or both.
            usecols:
                If only a subset of the fields specified in the resource's schema is to be loaded,
                the names or positions of the subset.

        Returns:
            The dataframe or DimcatResource.
        """
        dataframe = load_fl_resource(
            self._resource, index_col=index_col, usecols=usecols
        )
        return dataframe

    @cache
    def get_default_analysis(self) -> Rs:
        """Returns the default analysis of the resource."""
        return self.apply_step(self._default_analyzer)

    def get_grouping_levels(
        self, smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE
    ) -> List[str]:
        """Returns the levels of the grouping index, i.e., all levels until and including 'piece' or 'slice'."""
        smallest_unit = UnitOfAnalysis(smallest_unit)
        if smallest_unit == UnitOfAnalysis.SLICE:
            return self.get_level_names()[:-1]
        if smallest_unit == UnitOfAnalysis.PIECE:
            return self.get_piece_index(max_levels=0).names
        if smallest_unit == UnitOfAnalysis.GROUP:
            return self.default_groupby

    def get_index(self) -> DimcatIndex:
        """Returns the index of the resource based on the ``primaryKey`` of the :obj:`frictionless.Schema`."""
        return DimcatIndex.from_resource(self)

    def get_level_names(self) -> List[str]:
        """Returns the level names of the resource's index."""
        return self.get_index().names

    def get_normpath(
        self,
        set_default_if_missing=False,
    ) -> str:
        try:
            return self.normpath
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            return os.path.join(
                self.get_basepath(set_default_if_missing=set_default_if_missing),
                self.get_filepath(set_default_if_missing=set_default_if_missing),
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

    @cache
    def get_slice_intervals(
        self, round: Optional[int] = None, level_name: Optional[str] = None
    ) -> SliceIntervals:
        """Returns a :class:`SliceIntervals` object based on the result of :meth:`get_time_spans`.
        Effectively, this is this resource's :class:`DimcatIndex` with an appended level containing
        the time spans of the events represented by the resource's rows. This object can be used to
        slice any other resource that has pieces in common.
        """
        time_spans = self.get_time_spans(round=round, to_float=True, dropna=True)
        relevant_subset = time_spans[["start", "end"]]
        index_tuples, erroneous = [], []
        if level_name is None:
            level_name = f"{self.name.lower()}_slice"
        level_names = list(time_spans.index.names[:-1]) + [level_name]
        for idx, start, end in relevant_subset.itertuples(index=True, name=None):
            if end < start:
                erroneous.append(idx)
                continue
            interval = pd.Interval(start, end, closed="left")
            idx_tuple = idx[:-1] + (interval,)
            index_tuples.append(idx_tuple)
        return SliceIntervals.from_tuples(index_tuples, level_names=level_names)

    def get_time_spans(
        self, round: Optional[int] = None, to_float: bool = True, dropna: bool = False
    ) -> D:
        """Returns a dataframe with start ('left') and end ('end') positions of the events represented by this
        resource's rows.

        Args:
            round:
                To how many decimal places to round the intervals' boundary values. Setting a value automatically sets
                ``to_float=True``.
            to_float: Set to True to turn the time span values into floats.

        Returns:

        """
        df = self.df
        if "quarterbeats_all_endings" in df.columns:
            start_col = "quarterbeats_all_endings"
            if "quarterbeats" in df.columns:
                has_nan = df[start_col].isna().any()
                has_empty_strings = df[start_col].eq("").any()
                if has_nan or has_empty_strings:
                    df = df.copy()
                    if has_nan:
                        df[start_col].fillna(df["quarterbeats"], inplace=True)
                    if has_empty_strings:
                        df[start_col].where(
                            df[start_col].ne(""), df["quarterbeats"], inplace=True
                        )
        else:
            start_col = "quarterbeats"
        self.logger.debug(
            f"Using column {start_col!r} as for the left side of the computed time spans."
        )
        return get_time_spans_from_resource_df(
            df=df,
            start_column_name=start_col,
            duration_column_name="duration_qb",
            round=round,
            to_float=to_float,
            dropna=dropna,
            logger=self.logger,
        )

    def load(self, force_reload: bool = False) -> None:
        """Tries to load the data from disk into RAM. If successful, the .is_loaded property will be True.
        If the resource hadn't been loaded before, its .status property will be updated.
        """
        if not self.is_loaded or force_reload:
            _ = self.df

    def make_bar_plot(
        self,
        *step: StepSpecs,
        **kwargs,
    ) -> go.Figure:
        """Returns a plotly figure based on the default analysis or the analysis resulting from the given steps.

        Args:
            step:
                Zero or more PipelineSteps where the last one needs to return an object that has a .make_bar_plot()
                method, typically an :class:`Analyzer` returning a :class:`Result`. Defaults to
                :meth:`get_default_analysis` if no step is specified.
            **kwargs: Keyword arguments passed on to .make_bar_plot().

        Returns:
            The figure generated by calling .make_bar_plot() on the last step's result.
        """
        if not step:
            result = self.get_default_analysis()
        else:
            result = self.apply_step(*step)
        return result.make_bar_plot(**kwargs)

    def make_bubble_plot(
        self,
        *step: StepSpecs,
        **kwargs,
    ) -> go.Figure:
        """Returns a plotly figure based on the default analysis or the analysis resulting from the given steps.

        Args:
            step:
                Zero or more PipelineSteps where the last one needs to return an object that has a
                .make_bubble_plot() method, typically an :class:`Analyzer` returning a :class:`Result`. Defaults to
                :meth:`get_default_analysis` if no step is specified.
            **kwargs: Keyword arguments passed on to .make_bubble_plot().

        Returns:
            The figure generated by calling .make_bubble_plot() on the last step's result.
        """
        if not step:
            result = self.get_default_analysis()
        else:
            result = self.apply_step(*step)
        return result.make_bubble_plot(**kwargs)

    def make_pie_chart(
        self,
        *step: StepSpecs,
        **kwargs,
    ) -> go.Figure:
        """Returns a plotly figure based on the default analysis or the analysis resulting from the given steps.

        Args:
            step:
                Zero or more PipelineSteps where the last one needs to return an object that has a .make_pie_chart()
                method, typically an :class:`Analyzer` returning a :class:`Result`. Defaults to
                :meth:`get_default_analysis` if no step is specified.
            **kwargs: Keyword arguments passed on to .make_pie_chart().

        Returns:
            The figure generated by calling .make_pie_chart() on the last step's result.
        """
        if not step:
            result = self.get_default_analysis()
        else:
            result = self.apply_step(*step)
        return result.make_pie_chart(**kwargs)

    def _make_empty_fl_resource(self):
        """Create an empty frictionless resource object with a minimal descriptor."""
        return make_tsv_resource()

    def _prepare_feature_df(self, feature_config: DimcatConfig) -> D:
        """Prepare this resources dataframe for the extraction of a feature. This frequently involves subselecting
        relevant columns.
        """
        return self.df

    def plot(
        self,
        *step: StepSpecs,
        **kwargs,
    ) -> go.Figure:
        """Returns a plotly figure based on the default analysis or the analysis resulting from the given steps.

        Args:
            step:
                Zero or more PipelineSteps where the last one needs to return an object that has a .plot() method,
                typically an :class:`Analyzer` returning a :class:`Result`. Defaults to
                :meth:`get_default_analysis` if no step is specified.
            **kwargs: Keyword arguments passed on to .plot().

        Returns:
            The figure generated by calling .plot() on the last step's result.
        """
        if not step:
            result = self.get_default_analysis()
        else:
            result = self.apply_step(*step)
        return result.plot(**kwargs)

    def plot_grouped(self, *step: StepSpecs, **kwargs) -> go.Figure:
        """Returns a plotly figure based on the default analysis or the analysis resulting from the given steps.

        Args:
            step:
                Zero or more PipelineSteps where the last one needs to return an object that has a .plot_grouped()
                method, typically an :class:`Analyzer` returning a :class:`Result`. Defaults to
                :meth:`get_default_analysis` if no step is specified.
            **kwargs:
                Keyword arguments passed on to .plot_grouped().

        Returns:
            The figure generated by calling .plot_grouped() on the last step's result.
        """
        if not step:
            result: Result = self.get_default_analysis()
        else:
            result: Result = self.apply_step(*step)
        return result.plot_grouped(**kwargs)

    def set_basepath(
        self,
        basepath: str,
        reconcile: bool = False,
    ) -> None:
        super().set_basepath(
            basepath=basepath,
            reconcile=reconcile,
        )
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    def _set_dataframe(self, df: D):
        """Sets the dataframe without prior checks and assuming that it can be mutated safely, i.e. it is a copy."""
        df = self._format_dataframe(df)
        self._df = df
        if not self.column_schema.fields:
            try:
                self.column_schema = infer_schema_from_df(df)
            except FrictionlessException:
                self.logger.error(f"Could not infer schema from {type(df)}:\n{df}")
                raise
        else:
            try:
                self.validate(raise_exception=True)
            except FrictionlessException as e:
                raise DataframeIncompatibleWithColumnSchemaError(
                    self.resource_name, e, self.field_names, df.columns
                )
        self._update_status()

    def set_dataframe(self, df):
        if self.descriptor_exists:
            # ToDo: Enable creating new, date-based descriptor name for new Resources
            raise PotentiallyUnrelatedDescriptorError(
                message=f"Cannot set dataframe on a resource the points to the existing descriptor file at "
                f"{self.get_descriptor_path()}, because that "
                f"could lead to a discrepancy between the dataframe and the descriptor."
                f"Maybe you want to create a new resource via {self.name}.from_dataframe(<dataframe>)?"
            )
        if self.resource_exists:
            raise ResourceIsFrozenError(
                message=f"Cannot set dataframe on a resource {self.resource_name} that's pointing to an existing "
                f"resource {self.normpath}. "
            )
        if self.is_loaded:
            raise RuntimeError("This resource already includes a dataframe.")
        if isinstance(df, DimcatResource):
            df = df.df.copy()
        elif isinstance(df, pd.Series):
            df = df.to_frame()
            self.logger.info(
                f"Got a series, converted it into a dataframe with column name {df.columns[0]}."
            )
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            raise TypeError(f"Expected pandas.DataFrame, got {type(df)!r}.")
        self._set_dataframe(df)
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

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

    def store_dataframe(self, overwrite=False, validate: bool = True) -> None:
        """Stores the dataframe and its descriptor to disk based on the resource's configuration.

        Args:
            overwrite:
            validate:

        Raises:
            RuntimeError: If the resource is frozen or does not contain a dataframe or if the file exists already.
        """
        full_path = self.get_normpath(set_default_if_missing=True)
        if not overwrite and self.resource_exists:
            FileExistsError(
                f"Pass overwrite=True if you want to overwrite the existing {full_path}"
            )
        if self.status < ResourceStatus.DATAFRAME:
            raise RuntimeError(f"This {self.name} does not contain a dataframe.")
        ms3.write_tsv(self.df.reset_index(), full_path)
        self.logger.info(f"{self.name} serialized to {full_path}.")
        self.store_descriptor()
        if validate:
            report = self.validate(raise_exception=False)
            if report.valid:
                self.logger.info(f"Resource stored to {full_path} and validated.")
            else:
                errors = "\n".join(
                    str(err.message) for task in report.tasks for err in task.errors
                )
                msg = f"The resource did not validate after being stored to {full_path}:\n{errors}"
                if get_setting("never_store_unvalidated_data"):
                    os.remove(full_path)
                    self.logger.info(
                        msg
                        + "\nThe file was deleted because of the 'never_store_unvalidated_data' setting."
                    )
                self.logger.warning(msg)
        if self.status != ResourceStatus.STANDALONE_LOADED:
            status_before = self.status
            self._status = ResourceStatus.STANDALONE_LOADED
            resource_status_logger.debug(
                f"After writing {self.resource_name} to disk, the status has been changed from {status_before!r} to "
                f"{self.status!r}"
            )

    def _transform_feature_df(self, feature_df: D, feature_config: DimcatConfig) -> D:
        """Transform the dataframe after it has been prepared for feature extraction. This frequently involves
        dropping rows."""
        return feature_df

    def _sort_columns(self, df: D) -> D:
        """Sort the columns of the given dataframe in the order specified by :meth:`get_default_column_names` which
        combines the context columns with the class variabls :attr:`_auxiliary_column_names`,
        :attr:`_convenience_column_names`, and :attr:`_feature_column_names`. If the latter is not specified,
        the dataframe is returned as is because the purpose of this method is to have the feature columns at the end.
        """
        if self._feature_column_names:
            column_order = [
                col for col in self.get_default_column_names() if col in df.columns
            ]
            df = df[column_order]
        return df

    def summary_dict(self) -> dict:
        summary = self.to_dict()
        summary["ResourceStatus"] = self.status.name
        return summary

    def update_default_groupby(self, new_level_name: str) -> None:
        """Updates the value of :attr:`default_groupby` by prepending the new level name to it."""
        current_default = self.default_groupby
        if len(current_default) == 0:
            self.logger.debug(f"Default grouping level set to {new_level_name!r}.")
            new_default_value = [new_level_name]
        elif current_default[0] == new_level_name:
            self.logger.debug(
                f"Default groupby levels already start with {new_level_name!r}: {current_default}."
            )
            new_default_value = current_default
        elif new_level_name in current_default:
            new_default_value = [new_level_name] + [
                level for level in current_default if level != new_level_name
            ]
            self.logger.debug(
                f"Default groupby levels already contained {new_level_name!r}, so it was moved to the first position: "
                f"{new_default_value!r}."
            )
        else:
            new_default_value = [new_level_name] + current_default
            self.logger.debug(
                f"Updating default levels from {current_default} to {new_default_value}."
            )
        self.default_groupby = new_default_value

    def validate(
        self,
        raise_exception: bool = False,
        only_if_necessary: bool = False,
    ) -> Optional[fl.Report]:
        """Validate the resource's data against its descriptor.

        Args:
            raise_exception: (default False) Pass True to raise if the resource is not valid.
            only_if_necessary:
                (default False) Pass True to skip validation if the resource has already been validated or is
                assumed to be valid because it exists on disk.

        Returns:
            None if no validation took place (e.g. because resource is empty or ``only_if_necessary`` was True).
            Otherwise, frictionless report resulting from validating the data against the :attr:`column_schema`.

        Raises:
            FrictionlessException: If the resource is not valid and ``raise_exception`` is True.
        """
        if self.is_empty:
            self.logger.info("Nothing to validate.")
            return
        if only_if_necessary and (
            self._is_valid is not None or self.status >= ResourceStatus.VALIDATED
        ):
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
                status_before = self.status
                self._status = ResourceStatus.VALIDATED
                resource_status_logger.debug(
                    f"After successful validation, the status of {self.resource_name!r} has been changed from "
                    f"{status_before!r} to {self.status!r}"
                )
        else:
            errors = [err.message for task in report.tasks for err in task.errors]
            if self.status == ResourceStatus.VALIDATED:
                status_before = self.status
                self._status = ResourceStatus.DATAFRAME
                resource_status_logger.debug(
                    f"After unsuccessful validation, the status of {self.resource_name!r} has been changed from "
                    f"{status_before!r} to {self.status!r}"
                )
            if get_setting("never_store_unvalidated_data") and raise_exception:
                raise fl.FrictionlessException("\n".join(errors))
        return report

    def _resolve_group_cols_arg(
        self, group_cols: Optional[UnitOfAnalysis | str | Iterable[str]]
    ) -> List[str]:
        if not group_cols:
            groupby = []
        elif isinstance(group_cols, str):
            try:
                u_o_a = UnitOfAnalysis(group_cols)
            except ValueError:
                u_o_a = None
            if u_o_a is None:
                groupby = [group_cols]
            else:
                groupby = self.get_grouping_levels(u_o_a)
        else:
            groupby = list(group_cols)
        return groupby


# endregion DimcatResource
# region DimcatIndex


class IndexField(mm.fields.Field):
    """A marshmallow field for :obj:`DimcatIndex` objects."""

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

    class PickleSchema(Data.Schema):
        index = IndexField(allow_none=True)
        names = mm.fields.List(mm.fields.Str(), allow_none=True)

        @mm.post_load
        def init_object(self, data, **kwargs) -> DimcatIndex:
            index_value = data["index"]
            if isinstance(index_value, dict):
                raise NotImplementedError(index_value)
            if isinstance(index_value, pd.MultiIndex):
                return DimcatIndex(index_value)
            if isinstance(index_value, DimcatIndex):
                return index_value
            # should be an iterable of tuples
            if "names" not in data:
                raise mm.ValidationError(
                    f"When deserializing from {type(index_value)}, 'names' must be specified."
                )
            dtype = data.get("dtype", "DimcatIndex")
            Constructor = get_class(dtype)
            return Constructor.from_tuples(index_value, level_names=data.get("names"))

    class Schema(PickleSchema, Data.Schema):
        pass

    @classmethod
    def from_dataframe(cls, df: SomeDataframe) -> Self:
        """Create a DimcatIndex from a dataframe's index."""
        return cls.from_index(df.index)

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[Hashable, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a DimcatIndex from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set False to enable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouping = make_index_from_grouping_dict(
            grouping=grouping,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls.from_index(grouping, max_levels=0)

    @classmethod
    def from_index(cls, index: SomeIndex, **kwargs) -> Self:
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

    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple],
        level_names: Sequence[str],
    ) -> Self:
        list_of_tuples = list(tuples)
        if len(list_of_tuples) == 0:
            return cls(pd.MultiIndex.from_tuples([], names=level_names))
        first_tuple = list_of_tuples[0]
        if len(first_tuple) != len(level_names):
            raise ValueError(
                f"Expected tuples of length {len(level_names)}, got {len(first_tuple)}."
            )
        multiindex = pd.MultiIndex.from_tuples(list_of_tuples, names=level_names)
        return cls(multiindex)

    def __init__(
        self,
        index: Optional[IX] = None,
        basepath: Optional[str] = None,
    ):
        super().__init__(basepath=basepath)
        if index is None:
            self._index = pd.MultiIndex.from_tuples([], names=["corpus", "piece"])
        elif isinstance(index, pd.MultiIndex):
            if None in index.names:
                raise ValueError("Index cannot have a None name: {index.names}.")
            for name in index.names:
                check_name(name)
            self._index = index.copy()
        else:
            raise TypeError(f"Expected None or pandas.MultiIndex, got {type(index)!r}.")

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
        try:
            return getattr(self._index, item)
        except AttributeError:
            raise AttributeError(
                f"AttributeError: {self.name!r} object has no attribute {item!r}"
            )

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

    def filter(
        self,
        keep_values: levelvalue_ | Iterable[levelvalue_] = None,
        drop_values: levelvalue_ | Iterable[levelvalue_] = None,
        level: int | str = 0,
        drop_level: Optional[bool] = None,
    ) -> Self:
        """Returns a copy of the index with only those items where the given level has wanted values.

        Args:
            keep_values:
                One or several values to keep (dropping the rest). If a value is specified both for keeping and
                dropping, it is dropped.
            drop_values: One or several values to drop.
            level: Which index level to filter on.
            drop_level:
                Boolean specifies whether to keep the filtered level or to drop it. The default (None) corresponds
                to automatic behaviour, where the level is dropped if only one value remains, otherwise kept.

        Returns:
            A copy of the index with only those items where the given level has wanted values and may have been removed.
        """
        if not isinstance(level, (int, str)):
            raise TypeError(
                f"Level must be an int position or name string, got {type(level)!r}."
            )
        drop_this, keep_values = self.get_level_values_to_drop(
            drop_values, keep_values, level
        )
        new_index = self.index.drop(tuple(drop_this), level=level, errors="ignore")
        if drop_level or (drop_level is None and len(keep_values) == 1):
            new_index = new_index.droplevel(level)
        return self.__class__(new_index)

    def get_level_values_to_drop(
        self,
        drop_values: levelvalue_ | Iterable[levelvalue_],
        keep_values: levelvalue_ | Iterable[levelvalue_],
        level: int | str,
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        level_ints = resolve_levels_argument(level, self.names)
        assert (
            len(level_ints) == 1
        ), f"Level argumented should have resolved to a single integer, got {level_ints}."
        level_int = level_ints[0]
        level_values = set(self._index.levels[level_int])
        if drop_values is None:
            drop_this = set()
        elif isinstance(
            drop_values, (str, Number, bool)
        ):  # types = levelvalue_ TypeAlias
            drop_this = {drop_values}
        else:
            drop_this = set(drop_values)
        not_valid = drop_this.difference(level_values)
        if len(not_valid) > 0:
            self.logger.warning(
                f"The following drop_values are not present on level {level}: {not_valid}."
            )
            drop_this = drop_this.difference(not_valid)
        if keep_values:
            if isinstance(keep_values, (str, Number, bool)):
                keep_values = {keep_values}
            else:
                keep_values = set(keep_values)
            drop_this.update(level_values.difference(keep_values))
        keep_values = level_values.difference(drop_this)
        return drop_this, keep_values

    def sample(self, n: int) -> Self:
        """Return a random sample of n elements."""
        as_series = self._index.to_series()
        sample = as_series.sample(n)
        as_index = pd.MultiIndex.from_tuples(sample, names=self.names)
        return self.__class__(as_index)

    def to_resource(self, **kwargs) -> DimcatResource:
        """Create a DimcatResource from this index."""
        return DimcatResource.from_index(self, **kwargs)


class SliceIntervals(DimcatIndex):
    pass


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
        if max_levels > 0 and piece_level_position >= max_levels:
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

    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple],
        level_names: Sequence[str] = ("corpus", "piece"),
    ) -> Self:
        return super().from_tuples(tuples, level_names)

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
# region Feature


FIFTH_FEATURE_NAMES = (FeatureName.BassNotes, FeatureName.Notes)
HARMONY_FEATURE_NAMES = (
    FeatureName.BassNotes,
    FeatureName.HarmonyLabels,
    FeatureName.KeyAnnotations,
)


class Feature(DimcatResource):
    _enum_type = FeatureName

    def __init__(
        self,
        format=None,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        self._format = None
        if format is not None:
            self.format = format

    @property
    def format(self) -> None:
        return self._format

    @format.setter
    def format(self, value):
        if value is not None:
            warnings.warn(
                f"Setting format for {self.name} is inconsequential because no setter has been defined.",
                RuntimeWarning,
            )

    def get_available_column_names(
        self,
        index_levels: bool = False,
        context_columns: bool = False,
        auxiliary_columns: bool = False,
        convenience_columns: bool = False,
        feature_columns: bool = False,
    ):
        """Returns the column names that are available on the resource."""
        column_names = []
        if context_columns:
            column_names.extend(get_setting("context_columns"))
        if auxiliary_columns and self._auxiliary_column_names:
            column_names.extend(self._auxiliary_column_names)
        if convenience_columns and self._convenience_column_names:
            column_names.extend(self._convenience_column_names)
        if feature_columns and self._feature_column_names:
            column_names.extend(self._feature_column_names)
        available_columns = [col for col in column_names if col in self.df.columns]
        if index_levels:
            available_columns = self.get_level_names() + available_columns
        return available_columns


FeatureSpecs: TypeAlias = Union[MutableMapping, Feature, FeatureName, str]

# endregion Feature
