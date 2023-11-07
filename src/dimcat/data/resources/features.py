from __future__ import annotations

import logging
from functools import cache, cached_property
from typing import (
    ClassVar,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Type,
    TypeAlias,
    Union,
)

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import (
    DimcatConfig,
    FriendlyEnum,
    ObjectEnum,
    get_class,
    get_setting,
    is_subclass_of,
)
from dimcat.data.resources.base import D, ResourceStatus
from dimcat.data.resources.dc import DimcatResource
from dimcat.data.resources.utils import (
    boolean_is_minor_column_to_mode,
    condense_dataframe_by_groups,
    ensure_level_named_piece,
    infer_schema_from_df,
    load_fl_resource,
    make_adjacency_groups,
    resolve_recognized_piece_columns_argument,
)
from dimcat.dc_exceptions import (
    ResourceIsMissingFeatureColumnError,
    ResourceNotProcessableError,
)

logger = logging.getLogger(__name__)


class FeatureName(ObjectEnum):
    Annotations = "Annotations"
    Articulation = "Articulation"
    BassNotes = "BassNotes"
    HarmonyLabels = "HarmonyLabels"
    KeyAnnotations = "KeyAnnotations"
    Measures = "Measures"
    Metadata = "Metadata"
    Notes = "Notes"

    def get_class(self) -> Type[Feature]:
        return get_class(self.name)


HARMONY_FEATURE_NAMES = (
    FeatureName.BassNotes,
    FeatureName.HarmonyLabels,
    FeatureName.KeyAnnotations,
)


class Feature(DimcatResource):
    _enum_type = FeatureName
    _auxiliary_columns: Optional[ClassVar[List[str]]] = None
    _feature_columns: Optional[ClassVar[List[str]]] = None
    """Feature columns should be able to fully define an individual object. When creating the resource, any row
    containing a missing value in one of the feature columns is dropped."""

    # region constructors

    # @classmethod
    # def from_descriptor(
    #         cls,
    #         descriptor: dict | fl.Resource,
    #         descriptor_filename: Optional[str] = None,
    #         basepath: Optional[str] = None,
    #         auto_validate: bool = False,
    #         default_groupby: Optional[str | list[str]] = None,
    # ) -> Self:
    #     """Create a Feature by loading its frictionless descriptor from disk.
    #     The descriptor's directory is used as ``basepath``. ``descriptor_path`` is expected to end in
    #     ``.resource.json``.
    #
    #     Args:
    #         descriptor: Descriptor corresponding to a frictionless resource descriptor.
    #         descriptor_filename:
    #             Relative filepath for using a different JSON/YAML descriptor filename than the default
    #             :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
    #             setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
    #         basepath: Where the file would be serialized.
    #         auto_validate:
    #             By default, the DimcatResource will not be validated upon instantiation or change (but always before
    #             writing to disk). Set True to raise an exception during creation or modification of the resource,
    #             e.g. replacing the :attr:`column_schema`.
    #         default_groupby:
    #             Pass a list of column names or index levels to groupby something else than the default (by piece).
    #     """
    #     return super().from_descriptor(
    #         descriptor=descriptor,
    #         descriptor_filename=descriptor_filename,
    #         basepath=basepath,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #     )
    #
    # @classmethod
    # def from_descriptor_path(
    #         cls,
    #         descriptor_path: str,
    #         auto_validate: bool = False,
    #         default_groupby: Optional[str | list[str]] = None,
    # ) -> Self:
    #     """Create a Resource from a frictionless descriptor file on disk.
    #
    #     Args:
    #         descriptor_path: Absolute path where the JSON/YAML descriptor is located.
    #         auto_validate:
    #             By default, the DimcatResource will not be validated upon instantiation or change (but always before
    #             writing to disk). Set True to raise an exception during creation or modification of the resource,
    #             e.g. replacing the :attr:`column_schema`.
    #         default_groupby:
    #             Pass a list of column names or index levels to groupby something else than the default (by piece).
    #
    #     """
    #     return super().from_descriptor_path(
    #         descriptor_path=descriptor_path,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #     )
    #
    # @classmethod
    # def from_dataframe(
    #         cls,
    #         df: D,
    #         resource_name: str,
    #         descriptor_filename: Optional[str] = None,
    #         basepath: Optional[str] = None,
    #         auto_validate: bool = False,
    #         default_groupby: Optional[str | list[str]] = None,
    # ) -> Self:
    #     """Create a Feature from a dataframe, specifying its name and, optionally, at what path it is to be
    #     serialized.
    #
    #     Args:
    #         df: Dataframe to create the resource from.
    #         resource_name:
    #             Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
    #             is stored to a ZIP file.
    #         basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
    #         auto_validate:
    #             By default, the DimcatResource will not be validated upon instantiation or change (but always before
    #             writing to disk). Set True to raise an exception during creation or modification of the resource,
    #             e.g. replacing the :attr:`column_schema`.
    #         default_groupby:
    #             Pass a list of column names or index levels to groupby something else than the default (by piece).
    #     """
    #     new_object = cls(
    #         basepath=basepath,
    #         descriptor_filename=descriptor_filename,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #     )
    #     if resource_name is not None:
    #         new_object.resource_name = resource_name
    #     new_object._df = df
    #     new_object._update_status()
    #     return new_object
    #
    # @classmethod
    # def from_filepath(
    #         cls,
    #         filepath: str,
    #         resource_name: Optional[str] = None,
    #         descriptor_filename: Optional[str] = None,
    #         auto_validate: bool = False,
    #         default_groupby: Optional[str | list[str]] = None,
    #         basepath: Optional[str] = None,
    #         **kwargs: Optional[bool],
    # ) -> Self:
    #     """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.
    #
    #     Args:
    #         filepath: Path pointing to a resource descriptor or a simple path resource.
    #         resource_name:
    #             Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
    #             is stored to a ZIP file.
    #         descriptor_filename:
    #             Relative filepath for using a different JSON/YAML descriptor filename than the default
    #             :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
    #             setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
    #         auto_validate:
    #             By default, the Resource will not be validated upon instantiation or change (but always before
    #             writing to disk). Set True to raise an exception during creation or modification of the resource,
    #             e.g. replacing the :attr:`column_schema`.
    #         default_groupby:
    #             Pass a list of column names or index levels to groupby something else than the default (by piece).
    #         basepath:
    #             Basepath to use for the resource. If None, the folder of the ``filepath`` is used.
    #     """
    #     return super().from_filepath(
    #         filepath=filepath,
    #         resource_name=resource_name,
    #         descriptor_filename=descriptor_filename,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #         basepath=basepath,
    #         **kwargs,
    #     )
    #
    # @classmethod
    # def from_index(
    #         cls,
    #         index: DimcatIndex | SomeIndex,
    #         resource_name: str,
    #         basepath: Optional[str] = None,
    #         descriptor_filename: Optional[str] = None,
    #         auto_validate: bool = False,
    #         default_groupby: Optional[str | list[str]] = None,
    # ) -> Self:
    #     if isinstance(index, DimcatIndex):
    #         index = index.index
    #     dataframe = pd.DataFrame(index=index)
    #     return cls.from_dataframe(
    #         df=dataframe,
    #         resource_name=resource_name,
    #         descriptor_filename=descriptor_filename,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #         basepath=basepath,
    #     )
    #
    # @classmethod
    # def from_resource(
    #         cls,
    #         resource: Resource,
    #         descriptor_filename: Optional[str] = None,
    #         resource_name: Optional[str] = None,
    #         basepath: Optional[str] = None,
    #         auto_validate: Optional[bool] = None,
    #         default_groupby: Optional[str | list[str]] = None,
    # ) -> Self:
    #     """Create a Feature from an existing :obj:`Resource`, specifying its name and,
    #     optionally, at what path it is to be serialized.
    #
    #     Args:
    #         resource: An existing :obj:`frictionless.Resource` or a filepath.
    #         resource_name:
    #             Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
    #             is stored to a ZIP file.
    #         basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
    #         auto_validate:
    #             By default, the DimcatResource will not be validated upon instantiation or change (but always before
    #             writing to disk). Set True to raise an exception during creation or modification of the resource,
    #             e.g. replacing the :attr:`column_schema`.
    #         default_groupby:
    #             Pass a list of column names or index levels to groupby something else than the default (by piece).
    #     """
    #     if not isinstance(resource, Resource):
    #         raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
    #     new_object = super().from_resource(
    #         resource=resource,
    #         descriptor_filename=descriptor_filename,
    #         resource_name=resource_name,
    #         basepath=basepath,
    #         auto_validate=auto_validate,
    #         default_groupby=default_groupby,
    #     )
    #     # copy additional fields
    #     for attr in ("_df", "_status", "_corpus_name"):
    #         if (
    #                 hasattr(resource, attr)
    #                 and (value := getattr(resource, attr)) is not None
    #         ):
    #             setattr(new_object, attr, value)
    #     return new_object
    #
    # @classmethod
    # def from_resource_path(
    #         cls,
    #         resource_path: str,
    #         resource_name: Optional[str] = None,
    #         descriptor_filename: Optional[str] = None,
    #         **kwargs,
    # ) -> Self:
    #     if not resource_path.endswith(".tsv"):
    #         fname, fext = os.path.splitext(os.path.basename(resource_path))
    #         raise NotImplementedError(
    #             f"{fname}: Don't know how to load {fext} files yet."
    #             f"Either load the resource yourself and use {cls.name}.from_dataframe() or, if you "
    #             f"want to get a simple path resource, use Resource.from_resource_path() (not "
    #             f"DimcatResource)."
    #         )
    #     df = ms3.load_tsv(resource_path)
    #     return cls.from_dataframe(
    #         df=df,
    #         resource_name=resource_name,
    #         descriptor_filename=descriptor_filename,
    #         **kwargs,
    #     )

    # endregion constructors

    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        **kwargs,
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
            **kwargs: Keyword arguments passed to :meth:`_init_feature`.
        """
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        self._context_column_names: List[str] = []
        """Context columns present in this resource. Depend on the setting 'context_columns'."""
        self._feature_column_names: List[str] = []
        """Feature columns present in this resource."""
        self._treat_columns()
        self._modify_name()

    @property
    def auxiliary_column_names(self):
        if self._auxiliary_columns is None:
            return []
        return list(self._auxiliary_columns)

    @property
    def context_column_names(self) -> List[str]:
        return list(self._context_column_names)

    @cached_property
    def df(self) -> D:
        if self._df is not None:
            feature_df = self._df
        elif self.is_frozen:
            resource_df = self.get_dataframe()
            feature_df = self._make_feature_df(resource_df)
            self._resource.schema = infer_schema_from_df(
                feature_df
            )  # ToDo: the new schema should be attributed via
            # self.column_schema = ... but for that, the detachment from the feature from the original resource needs
            # to be implemented, which involves adapting the status
        else:
            RuntimeError(f"No dataframe accessible for this {self.name}:\n{self}")
        return feature_df

    @property
    def feature_column_names(self) -> List[str]:
        """List of column names that this feature uses."""

        if self._feature_columns is None:
            available_columns = [
                col
                for col in self.field_names
                if col not in self.column_schema.primary_key
            ]
            excluded_columns = list(self._context_column_names)
            if self._auxiliary_columns is not None:
                excluded_columns += self._auxiliary_columns
            return [col for col in available_columns if col not in excluded_columns]
        return list(self._feature_columns)

    @property
    def value_column(self) -> str:
        """The name of the column that contains the values of the resource. May depend on format
        settings.
        """
        if self._value_column is not None:
            return self._value_column
        if self.default_value_column is not None:
            return self.default_value_column
        if self._feature_columns is not None:
            return self._feature_columns[-1]
        return self.column_schema.field_names[-1]

    @value_column.setter
    def value_column(self, value_column: str):
        if not isinstance(value_column, str):
            raise TypeError(f"Expected a string, got {type(value_column)}")
        if value_column not in self.field_names:
            raise ValueError(
                f"Column {value_column!r} does not exist in the resource's schema."
            )
        self._value_column = value_column

    def get_column_names(self, include_index_levels: bool = False) -> List[str]:
        """Retrieve the names of [index_levels] + auxiliary_column_names + feature_column_names"""
        column_names = self.column_schema.primary_key if include_index_levels else []
        column_names += (
            self.context_column_names
            + self.auxiliary_column_names
            + self.feature_column_names
        )
        return column_names

    @cache
    def get_dataframe(self) -> D:
        """
        Load the dataframe from disk based on the descriptor's normpath.

        Returns:
            The dataframe or DimcatResource.
        """
        index_levels = self.column_schema.primary_key
        usecols = self.get_column_names()
        dataframe = load_fl_resource(
            self._resource, index_col=index_levels, usecols=usecols
        )
        if not index_levels:
            dataframe.index.rename("i", inplace=True)
            recognized_piece_columns = resolve_recognized_piece_columns_argument()
            if not any(col in dataframe.columns for col in recognized_piece_columns):
                piece_name = self.filepath.split(".")[0]
                dataframe = pd.concat([dataframe], keys=[piece_name], names=["piece"])
        if "piece" not in dataframe.index.names:
            dataframe.index, _ = ensure_level_named_piece(dataframe.index)
        if self.status == ResourceStatus.STANDALONE_NOT_LOADED:
            self._status = ResourceStatus.STANDALONE_LOADED
        elif self.status == ResourceStatus.PACKAGED_NOT_LOADED:
            self._status = ResourceStatus.PACKAGED_LOADED
        return dataframe

    def _make_feature_df(
        self,
        resource_df: D,
    ):
        len_before = len(resource_df)
        feature_df = self._transform_resource_df(resource_df)
        columns = self.get_column_names()
        feature_df = feature_df[columns]
        len_after = len(feature_df)
        if len_before == len_after:
            self.logger.debug(
                f"Made {self.dtype} dataframe for {self.resource_name} with {len_after} non-empty rows."
            )
        else:
            self.logger.debug(
                f"Made {self.dtype} dataframe for {self.resource_name} dropping {len_before - len_after} "
                f"rows with missing values."
            )
        return feature_df

    def _modify_name(self):
        """Modify the :attr:`resource_name` to reflect the feature."""
        pass

    def _transform_resource_df(self, feature_df):
        """Called by :meth:`_make_feature_df` to transform the resource dataframe into a feature dataframe."""
        if self._feature_columns is None:
            result = feature_df.copy()
        else:
            result = feature_df.dropna(subset=self._feature_columns, how="any")
        return result

    def _treat_columns(self) -> None:
        """Check which columns exist in the original resource and store the one that this feature uses."""
        if self.is_empty:
            return
        available_columns = self.field_names
        assert len(self.field_names), "No column schema defined for the given resource."
        context_column_names = get_setting("context_columns")
        if self._feature_columns is not None:
            missing = [
                col for col in self._feature_columns if col not in available_columns
            ]
            if missing:
                raise ResourceIsMissingFeatureColumnError(
                    self.resource_name, missing, self.name
                )
            # make sure context columns do not include feature columns
            context_column_names = [
                col for col in context_column_names if col not in self._feature_columns
            ]
        self._context_column_names = [
            col for col in context_column_names if col in available_columns
        ]


class Metadata(Feature):
    pass


# region Annotations


class Annotations(Feature):
    pass


class HarmonyLabelsFormat(FriendlyEnum):
    ROMAN = "ROMAN"
    FIFTHS = "FIFTHS"
    SCALE_DEGREE = "SCALE_DEGREE"
    INTERVAL = "INTERVAL"


class HarmonyLabels(Annotations):
    _auxiliary_columns = [
        "globalkey",
        "localkey",
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
        "root_roman",
        "chord_and_mode",
    ]
    _extractable_features = HARMONY_FEATURE_NAMES
    default_value_column = "chord_and_mode"

    class Schema(Annotations.Schema):
        format = mm.fields.Enum(HarmonyLabelsFormat)

    def __init__(
        self,
        format: HarmonyLabelsFormat = HarmonyLabelsFormat.ROMAN,
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
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        self._format = HarmonyLabelsFormat(format)

    @property
    def format(self) -> HarmonyLabelsFormat:
        return self._format

    def _transform_resource_df(self, feature_df):
        """Called by :meth:`_make_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = super()._transform_resource_df(feature_df)
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_harmony_feature(feature_df)
        return feature_df


def safe_row_tuple(row):
    try:
        return ", ".join(row)
    except TypeError:
        return pd.NA


def extend_keys_feature(
    feature_df,
):
    concatenate_this = [
        feature_df,
        boolean_is_minor_column_to_mode(feature_df.globalkey_is_minor).rename(
            "globalkey_mode"
        ),
        boolean_is_minor_column_to_mode(feature_df.localkey_is_minor).rename(
            "localkey_mode"
        ),
        ms3.transform(
            feature_df, ms3.resolve_relative_keys, ["localkey", "localkey_is_minor"]
        ).rename("localkey_resolved"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    concatenate_this = [
        feature_df,
        feature_df[["localkey", "globalkey_mode"]]
        .apply(safe_row_tuple, axis=1)
        .rename("localkey_and_mode"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


def extend_harmony_feature(
    feature_df,
):
    """Requires previous application of transform_keys_feature."""
    concatenate_this = [
        feature_df,
        (feature_df.numeral + ("/" + feature_df.relativeroot).fillna("")).rename(
            "root_roman"
        ),
        ms3.transform(
            feature_df, ms3.resolve_relative_keys, ["pedal", "localkey_is_minor"]
        ).rename("pedal_resolved"),
        feature_df[["chord", "localkey_mode"]]
        .apply(safe_row_tuple, axis=1)
        .rename("chord_and_mode"),
        # ms3.transform(
        #     feature_df, ms3.rel2abs_key, ["numeral", "localkey_resolved", "localkey_resolved_is_minor"]
        # ).rename("root_roman_resolved"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class BassNotes(HarmonyLabels):
    _auxiliary_columns = [
        "globalkey",
        "localkey",
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
    ]
    _feature_columns = ["bass_note"]
    _extractable_features = None

    def _modify_name(self):
        """Modify the :attr:`resource_name` to reflect the feature."""
        self.resource_name = f"{self.resource_name}.bass_notes"


class KeyAnnotations(Annotations):
    _auxiliary_columns = [
        "globalkey_is_minor",
        "localkey_is_minor",
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
    ]
    _feature_columns = ["globalkey", "localkey"]
    _extractable_features = None
    default_value_column = "localkey_and_mode"

    def _transform_resource_df(self, feature_df):
        """Called by :meth:`_make_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = super()._transform_resource_df(feature_df)
        groupby_levels = feature_df.index.names[:-1]
        group_keys, _ = make_adjacency_groups(
            feature_df.localkey, groupby=groupby_levels
        )
        feature_df = condense_dataframe_by_groups(feature_df, group_keys)
        feature_df = extend_keys_feature(feature_df)
        return feature_df


# endregion Annotations
# region Controls


class Articulation(Feature):
    pass


# endregion Controls
# region Events


class NotesFormat(FriendlyEnum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    SCALE_DEGREE = "SCALE_DEGREE"
    INTERVAL = "INTERVAL"


class Notes(Feature):
    default_analyzer = "PitchClassVectors"
    default_value_column = "tpc"

    class Schema(Feature.Schema):
        format = mm.fields.Enum(NotesFormat)
        merge_ties = mm.fields.Boolean(
            load_default=True,
            metadata=dict(
                title="Merge tied notes",
                description="If set, notes that are tied together in the score are merged together, counting them "
                "as a single event of the corresponding length. Otherwise, every note head is counted.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = True,
        weight_grace_notes: float = 0.0,
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
        self._format = NotesFormat(format)
        self._merge_ties = bool(merge_ties)
        self._weight_grace_notes = float(weight_grace_notes)

    @property
    def format(self) -> NotesFormat:
        return self._format

    @property
    def merge_ties(self) -> bool:
        return self._merge_ties

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes


# endregion Events
# region Structure
class Measures(Feature):
    pass


# endregion Structure
# region helpers

FeatureSpecs: TypeAlias = Union[MutableMapping, Feature, FeatureName, str]


def feature_specs2config(feature: FeatureSpecs) -> DimcatConfig:
    """Converts a feature specification into a dimcat configuration.

    Raises:
        TypeError: If the feature cannot be converted to a dimcat configuration.
    """
    if isinstance(feature, DimcatConfig):
        feature_config = feature
    elif isinstance(feature, Feature):
        feature_config = feature.to_config()
    elif isinstance(feature, MutableMapping):
        feature_config = DimcatConfig(feature)
    elif isinstance(feature, str):
        feature_name = FeatureName(feature)
        feature_config = DimcatConfig(dtype=feature_name)
    else:
        raise TypeError(
            f"Cannot convert the {type(feature).__name__} {feature!r} to DimcatConfig."
        )
    if feature_config.options_dtype == "DimcatConfig":
        feature_config = DimcatConfig(feature_config["options"])
    if not is_subclass_of(feature_config.options_dtype, Feature):
        raise TypeError(
            f"DimcatConfig describes a {feature_config.options_dtype}, not a Feature: "
            f"{feature_config.options}"
        )
    return feature_config


def features_argument2config_list(
    features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
    allowed_features: Optional[Iterable[str | FeatureName]] = None,
) -> List[DimcatConfig]:
    if features is None:
        return []
    if isinstance(features, (MutableMapping, Feature, FeatureName, str)):
        features = [features]
    configs = []
    for specs in features:
        configs.append(feature_specs2config(specs))
    if allowed_features:
        allowed_features = [FeatureName(f) for f in allowed_features]
        for config in configs:
            if config.options_dtype not in allowed_features:
                raise ResourceNotProcessableError(config.options_dtype)
    return configs


# endregion helpers
