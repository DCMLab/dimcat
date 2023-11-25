from __future__ import annotations

import logging
from typing import Optional

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import FriendlyEnum
from dimcat.data.resources.base import D
from dimcat.data.resources.dc import HARMONY_FEATURE_NAMES, Feature
from dimcat.data.resources.utils import (
    boolean_is_minor_column_to_mode,
    condense_dataframe_by_groups,
    make_adjacency_groups,
)
from dimcat.dc_exceptions import (
    DataframeIsMissingExpectedColumnsError,
    FeatureIsMissingFormatColumnError,
)

logger = logging.getLogger(__name__)


class Metadata(Feature):
    pass


# region Annotations
AUXILIARY_HARMONYLABEL_COLUMNS = [
    "cadence",
    "label",
    "phraseend",
    "chord_tones",
    "chord_type",
    "figbass",
    "form",
    "numeral",
]
"""These columns are included in sub-features of HarmonyLabels to enable more means of investigation,
such as groupers."""

KEY_CONVENIENCE_COLUMNS = [
    "globalkey_is_minor",
    "globalkey_mode",
    "localkey_and_mode",
    "localkey_is_minor",
    "localkey_mode",
    "localkey_resolved",
]
"""These columns are computed by default for all Annotations that include keys, where global keys are given as note
names, and local keys are given as Roman numerals. In both cases, lowercase strings are interpreted as minor keys."""


def extend_keys_feature(
    feature_df,
):
    columns_to_add = (
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("localkey", "localkey_is_minor", "globalkey_is_minor")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
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


class Annotations(Feature):
    pass


class DcmlAnnotations(Annotations):
    _auxiliary_column_names = [
        "color",
        "color_a",
        "color_b",
        "color_g",
        "color_r",
    ]
    _convenience_column_names = [
        "added_tones",
        "bass_note",
        "cadence",
        "changes",
        "chord_tones",
        "chord_type",
        "figbass",
        "form",
        "globalkey_is_minor",
        "globalkey_mode",
        "localkey_and_mode",
        "localkey_is_minor",
        "localkey_mode",
        "localkey_resolved",
        "numeral",
        "pedal",
        "pedalend",
        "phraseend",
        "relativeroot",
        "root",
        "special",
    ]
    _feature_column_names = ["label"]
    _default_value_column = "label"
    _extractable_features = HARMONY_FEATURE_NAMES

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = super()._format_dataframe(feature_df)
        feature_df = extend_keys_feature(feature_df)
        return feature_df


def extend_harmony_feature(
    feature_df,
):
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = ("root_roman", "pedal_resolved", "chord_and_mode")
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
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


class HarmonyLabelsFormat(FriendlyEnum):
    ROMAN = "ROMAN"
    FIFTHS = "FIFTHS"
    SCALE_DEGREE = "SCALE_DEGREE"
    INTERVAL = "INTERVAL"


class HarmonyLabels(DcmlAnnotations):
    _auxiliary_column_names = DcmlAnnotations._auxiliary_column_names + [
        "cadence",
        "label",
        "phraseend",
    ]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "added_tones",
        "bass_note",
        "changes",
        "chord_and_mode",
        "chord_tones",
        "chord_type",
        "figbass",
        "form",
        "numeral",
        "pedal",
        "pedalend",
        "relativeroot",
        "root",
        "special",
    ]
    _feature_column_names = [
        "globalkey",
        "localkey",
        "chord",
    ]
    _default_value_column = "chord_and_mode"

    class Schema(DcmlAnnotations.Schema):
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

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_harmony_feature(feature_df)
        return feature_df


def safe_row_tuple(row):
    try:
        return ", ".join(row)
    except TypeError:
        return pd.NA


def extend_bass_notes_feature(
    feature_df,
):
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = (
        "bass_note_over_local_tonic",
        "bass_degree",
        "bass_degree_and_mode",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("bass_note", "localkey_is_minor", "localkey_mode")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [
        feature_df,
        ms3.transform(feature_df.bass_note, ms3.fifths2iv).rename(
            "bass_note_over_local_tonic"
        ),
        ms3.transform(
            feature_df, ms3.fifths2sd, ["bass_note", "localkey_is_minor"]
        ).rename("bass_degree"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    concatenate_this = [
        feature_df,
        feature_df[["bass_degree", "localkey_mode"]]
        .apply(safe_row_tuple, axis=1)
        .rename("bass_degree_and_mode"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class BassNotesFormat(FriendlyEnum):
    FIFTHS = "FIFTHS"
    INTERVAL = "INTERVAL"
    SCALE_DEGREE = "SCALE_DEGREE"


class BassNotes(HarmonyLabels):
    _default_formatted_column = "bass_note_over_local_tonic"
    _default_value_column = "bass_note"
    _auxiliary_column_names = (
        DcmlAnnotations._auxiliary_column_names + AUXILIARY_HARMONYLABEL_COLUMNS
    )
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "bass_degree",
        "bass_note_over_local_tonic",
        "bass_degree_and_mode",
    ]
    _feature_column_names = [
        "globalkey",
        "localkey",
        "bass_note",
    ]
    _extractable_features = None

    class Schema(DcmlAnnotations.Schema):
        format = mm.fields.Enum(BassNotesFormat)

    def __init__(
        self,
        format: NotesFormat = BassNotesFormat.INTERVAL,
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
        self._format = BassNotesFormat.INTERVAL
        self.format = format

    @property
    def format(self) -> BassNotesFormat:
        return self._format

    @format.setter
    def format(self, format: BassNotesFormat):
        format = BassNotesFormat(format)
        if self.format == format:
            pass
        if format == BassNotesFormat.INTERVAL:
            new_formatted_column = "bass_note_over_local_tonic"
        elif format == BassNotesFormat.FIFTHS:
            new_formatted_column = "bass_note"
        elif format == BassNotesFormat.SCALE_DEGREE:
            if "mode" in self.get_default_groupby():
                new_formatted_column = "bass_degree"
            else:
                new_formatted_column = "bass_degree_and_mode"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    def _modify_name(self):
        """Modify the :attr:`resource_name` to reflect the feature."""
        self.resource_name = f"{self.resource_name}.bass_notes"

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_bass_notes_feature(feature_df)
        return feature_df


class KeyAnnotations(DcmlAnnotations):
    _auxiliary_column_names = ["label"]
    _convenience_column_names = [
        "globalkey_is_minor",
        "localkey_is_minor",
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
    ]
    _feature_column_names = ["globalkey", "localkey"]
    _extractable_features = None
    _default_value_column = "localkey_and_mode"

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe."""
        feature_df = extend_keys_feature(feature_df)
        groupby_levels = feature_df.index.names[:-1]
        group_keys, _ = make_adjacency_groups(
            feature_df.localkey, groupby=groupby_levels
        )
        feature_df = condense_dataframe_by_groups(feature_df, group_keys)
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
    _default_analyzer = "PitchClassVectors"
    _default_value_column = "tpc"

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


# endregion helpers
