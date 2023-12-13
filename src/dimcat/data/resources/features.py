from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import FriendlyEnum, FriendlyEnumField
from dimcat.data.resources.base import D, FeatureName
from dimcat.data.resources.dc import (
    HARMONY_FEATURE_NAMES,
    DimcatIndex,
    Feature,
    SliceIntervals,
    UnitOfAnalysis,
)
from dimcat.data.resources.results import tuple2str
from dimcat.data.resources.utils import (
    boolean_is_minor_column_to_mode,
    condense_dataframe_by_groups,
    join_df_on_index,
    make_adjacency_groups,
    merge_ties,
)
from dimcat.dc_exceptions import (
    DataframeIsMissingExpectedColumnsError,
    FeatureIsMissingFormatColumnError,
    ResourceIsMissingPieceIndexError,
)
from dimcat.utils import get_middle_composition_year

logger = logging.getLogger(__name__)


class Metadata(Feature):
    _default_analyzer = dict(dtype="Proportions", dimension_column="length_qb")
    _default_value_column = "piece"

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
        return join_df_on_index(self.df, slice_intervals)

    def get_composition_years(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
    ):
        group_cols = self._resolve_group_cols_arg(group_cols)
        years = get_middle_composition_year(metadata=self.df)
        if not group_cols:
            return years
        result = years.groupby(group_cols).mean()
        return result


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
    "chord",
    "root",
]
"""These columns are included in sub-features of HarmonyLabels to enable more means of investigation,
such as groupers."""

KEY_CONVENIENCE_COLUMNS = [
    "globalkey_is_minor",
    "localkey_is_minor",
    "globalkey_mode",
    "localkey_mode",
    "localkey_resolved",
    "localkey_and_mode",
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
    _convenience_column_names = (
        [
            "added_tones",
            "bass_note",
            "cadence",
            "changes",
            "chord",
            "chord_tones",
            "chord_type",
            "figbass",
            "form",
            "globalkey",
            "localkey",
        ]
        + KEY_CONVENIENCE_COLUMNS
        + [
            "numeral",
            "pedal",
            "pedalend",
            "phraseend",
            "relativeroot",
            "root",
            "special",
        ]
    )
    _feature_column_names = ["label"]
    _default_value_column = "label"
    _extractable_features = HARMONY_FEATURE_NAMES + (FeatureName.CadenceLabels,)

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = extend_keys_feature(feature_df)
        return self._sort_columns(feature_df)


def make_chord_col(df: D, cols: Optional[List[str]] = None, name: str = "chord"):
    """The 'chord' column contains the chord part of a DCML label, i.e. without indications of key, pedal, cadence, or
    phrase. This function can re-create this column, e.g. if the feature columns were changed. To that aim, the function
    takes a DataFrame and the column names that it adds together, creating new strings.
    """
    if cols is None:
        cols = ["numeral", "form", "figbass", "changes", "relativeroot"]
    cols = [c for c in cols if c in df.columns]
    summing_cols = [c for c in cols if c not in ("changes", "relativeroot")]
    if len(summing_cols) == 1:
        chord_col = df[summing_cols[0]].fillna("").astype("string")
    else:
        chord_col = df[summing_cols].fillna("").astype("string").sum(axis=1)
    if "changes" in cols:
        chord_col += ("(" + df.changes.astype("string") + ")").fillna("")
    if "relativeroot" in cols:
        chord_col += ("/" + df.relativeroot.astype("string")).fillna("")
    return chord_col.rename(name)


def extend_harmony_feature(
    feature_df,
):
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = (
        "root_roman",
        "pedal_resolved",
        "chord_and_mode",
        "chord_reduced",
        "chord_reduced_and_mode",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = (
        "chord",
        "form",
        "figbass",
        "pedal",
        "numeral",
        "relativeroot",
        "localkey_is_minor",
        "localkey_mode",
    )
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "root_roman" not in feature_df.columns:
        concatenate_this.append(
            (feature_df.numeral + ("/" + feature_df.relativeroot).fillna("")).rename(
                "root_roman"
            )
        )
    if "chord_reduced" not in feature_df.columns:
        concatenate_this.append(
            (
                reduced_col := make_chord_col(
                    feature_df,
                    cols=["numeral", "form", "figbass", "relativeroot"],
                    name="chord_reduced",
                )
            )
        )
    else:
        reduced_col = feature_df.chord_reduced
    if "chord_reduced_and_mode" not in feature_df.columns:
        concatenate_this.append(
            (reduced_col + ", " + feature_df.localkey_mode).rename(
                "chord_reduced_and_mode"
            )
        )
    if "pedal_resolved" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.resolve_relative_keys, ["pedal", "localkey_is_minor"]
            ).rename("pedal_resolved")
        )
    if "chord_and_mode" not in feature_df.columns:
        concatenate_this.append(
            feature_df[["chord", "localkey_mode"]]
            .apply(safe_row_tuple, axis=1)
            .rename("chord_and_mode")
        )
    # if "root_roman_resolved" not in feature_df.columns:
    #     concatenate_this.append(
    #         ms3.transform(
    #             feature_df,
    #             ms3.rel2abs_key,
    #             ["numeral", "localkey_resolved", "localkey_resolved_is_minor"],
    #         ).rename("root_roman_resolved")
    #     )
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


def chord_tones2interval_structure(
    fifths: Iterable[int], reference: Optional[int] = None
) -> Tuple[str]:
    """The fifth are interpreted as intervals expressing distances from the local tonic ("neutral degrees").
    The result will be a tuple of strings that express the same intervals but expressed with respect to the given
    reference (neutral degree), removing unisons.
    If no reference is specified, the first degree (usually, the bass note) is used as such.
    """
    try:
        fifths = tuple(fifths)
        if len(fifths) == 0:
            return ()
    except Exception:
        return ()
    if reference is None:
        reference = fifths[0]
    elif reference in fifths:
        position = fifths.index(reference)
        if position > 0:
            fifths = fifths[position:] + fifths[:position]
    adapted_intervals = [
        ms3.fifths2iv(adapted)
        for interval in fifths
        if (adapted := interval - reference) != 0
    ]
    return tuple(adapted_intervals)


def add_chord_tone_scale_degrees(
    feature_df,
):
    """Turns 'chord_tones' column into multiple scale-degree columns."""
    columns_to_add = (
        "scale_degrees",
        "scale_degrees_and_mode" "scale_degrees_major",
        "scale_degrees_minor",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("chord_tones", "localkey_is_minor", "localkey_mode")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "scale_degrees" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.fifths2sd, ["chord_tones", "localkey_is_minor"]
            ).rename("scale_degrees")
        )
    if "scale_degrees_major" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.chord_tones, ms3.fifths2sd, minor=False).rename(
                "scale_degrees_major"
            )
        )
    if "scale_degrees_minor" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.chord_tones, ms3.fifths2sd, minor=True).rename(
                "scale_degrees_minor"
            )
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    if "scale_degrees_and_mode" not in feature_df.columns:
        sd_and_mode = pd.Series(
            feature_df[["scale_degrees", "localkey_mode"]].itertuples(
                index=False, name=None
            ),
            index=feature_df.index,
            name="scale_degrees_and_mode",
        )
        concatenate_this = [feature_df, sd_and_mode.apply(tuple2str)]
        feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


def add_chord_tone_intervals(
    feature_df,
):
    """Turns 'chord_tones' column into one or two additional columns, depending on whether a 'root' column is
    present, where the chord_tones (which come as fifths) are represented as strings representing intervals over the
    bass_note and above the root, if present.
    """
    columns_to_add = (
        "intervals_over_bass",
        "intervals_over_root",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("chord_tones",)  # "root" is optional
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "intervals_over_bass" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df.chord_tones, chord_tones2interval_structure
            ).rename("intervals_over_bass")
        )
    if "intervals_over_root" not in feature_df.columns and "root" in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, chord_tones2interval_structure, ["chord_tones", "root"]
            ).rename("intervals_over_root")
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class HarmonyLabelsFormat(FriendlyEnum):
    """Format to display the chord labels in. ROMAN stands for Roman numerals, ROMAN_REDUCED for the same numerals
    without any suspensions, alterations, additions, etc."""

    ROMAN = "ROMAN"
    ROMAN_REDUCED = "ROMAN_REDUCED"
    SCALE_DEGREE = "SCALE_DEGREE"
    SCALE_DEGREE_MAJOR = "SCALE_DEGREE_MAJOR"
    SCALE_DEGREE_MINOR = "SCALE_DEGREE_MINOR"


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
        "chord_reduced",
        "chord_reduced_and_mode",
        "chord_tones",
        "scale_degrees",
        "scale_degrees_and_mode",
        "scale_degrees_major",
        "scale_degrees_minor",
        "intervals_over_bass",
        "intervals_over_root",
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
        format = FriendlyEnumField(
            HarmonyLabelsFormat,
            load_default=HarmonyLabelsFormat.ROMAN,
            metadata=dict(
                expose=True,
                description="Format to display the chord labels in. ROMAN stands for Roman numerals, ROMAN_REDUCED "
                "for the same numerals without any suspensions, alterations, additions, etc.",
            ),
        )

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
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @property
    def format(self) -> HarmonyLabelsFormat:
        return self._format

    @format.setter
    def format(self, format: HarmonyLabelsFormat):
        format = HarmonyLabelsFormat(format)
        if self.format == format:
            return
        if format == HarmonyLabelsFormat.ROMAN:
            new_formatted_column = "chord_and_mode"
        elif format == HarmonyLabelsFormat.ROMAN_REDUCED:
            new_formatted_column = "chord_reduced_and_mode"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE:
            new_formatted_column = "scale_degrees_and_mode"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE_MAJOR:
            new_formatted_column = "scale_degrees_major"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE_MINOR:
            new_formatted_column = "scale_degrees_minor"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def formatted_column(self) -> str:
        if self.format == HarmonyLabelsFormat.ROMAN:
            if "mode" in self.default_groupby:
                return "chord"
            else:
                return "chord_and_mode"
        elif self._format == HarmonyLabelsFormat.ROMAN_REDUCED:
            if "mode" in self.default_groupby:
                return "chord_reduced"
            else:
                return "chord_reduced_and_mode"
        elif self._format == HarmonyLabelsFormat.SCALE_DEGREE:
            if "mode" in self.default_groupby:
                return "scale_degrees"
            else:
                return "scale_degrees_and_mode"
        if self._formatted_column is not None:
            return self._formatted_column
        if self._default_formatted_column is not None:
            return self._default_formatted_column
        return

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_harmony_feature(feature_df)
        feature_df = add_chord_tone_intervals(feature_df)
        feature_df = add_chord_tone_scale_degrees(feature_df)
        return self._sort_columns(feature_df)


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
        "bass_degree_major",
        "bass_degree_minor",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("bass_note", "localkey_is_minor", "localkey_mode")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "bass_note_over_local_tonic" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2iv).rename(
                "bass_note_over_local_tonic"
            )
        )
    if "bass_degree" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.fifths2sd, ["bass_note", "localkey_is_minor"]
            ).rename("bass_degree")
        )
    if "bass_degree_major" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2sd, minor=False).rename(
                "bass_degree_major"
            )
        )
    if "bass_degree_minor" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2sd, minor=True).rename(
                "bass_degree_minor"
            )
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    if "bass_degree_and_mode" not in feature_df.columns:
        concatenate_this = [
            feature_df,
            feature_df[["bass_degree", "localkey_mode"]]
            .apply(safe_row_tuple, axis=1)
            .rename("bass_degree_and_mode"),
        ]
        feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class BassNotesFormat(FriendlyEnum):
    """Format to display the bass notes in. INTERVAL stands for the interval between the bass note and the local
    tonic, FIFTHS expresses that same interval as a number of fifths, SCALE_DEGREE expresses the bass note as a scale
    degree depending on the local key (i.e. scale degrees 3, 6, 7 are minor intervals in minor and major intervals in
    major), whereas SCALE_DEGREE_MAJOR and SCALE_DEGREE_MINOR express the bass note as a scale degree independent of
    the local key"""

    FIFTHS = "FIFTHS"
    INTERVAL = "INTERVAL"
    SCALE_DEGREE = "SCALE_DEGREE"
    SCALE_DEGREE_MAJOR = "SCALE_DEGREE_MAJOR"
    SCALE_DEGREE_MINOR = "SCALE_DEGREE_MINOR"


class BassNotes(HarmonyLabels):
    _default_formatted_column = "bass_note_over_local_tonic"
    _default_value_column = "bass_note"
    _auxiliary_column_names = (
        DcmlAnnotations._auxiliary_column_names + AUXILIARY_HARMONYLABEL_COLUMNS
    )
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "bass_degree",
        "bass_degree_and_mode",
        "bass_degree_major",
        "bass_degree_minor",
        "bass_note_over_local_tonic",
        "intervals_over_bass",
        "intervals_over_root",
        "scale_degrees",
        "scale_degrees_and_mode",
        "scale_degrees_major",
        "scale_degrees_minor",
    ]
    _feature_column_names = [
        "globalkey",
        "localkey",
        "bass_note",
    ]
    _extractable_features = None

    class Schema(DcmlAnnotations.Schema):
        format = FriendlyEnumField(
            BassNotesFormat,
            load_default=BassNotesFormat.INTERVAL,
            metadata=dict(
                expose=True,
                description="Format to display the bass notes in. INTERVAL stands for the interval between the bass "
                "note and the local tonic, FIFTHS expresses that same interval as a number of fifths, "
                "SCALE_DEGREE expresses the bass note as a scale degree depending on the local key (i.e. "
                "scale degrees 3, 6, 7 are minor intervals in minor and major intervals in major), "
                "whereas SCALE_DEGREE_MAJOR and SCALE_DEGREE_MINOR express the bass note as a scale "
                "degree independent of the local key",
            ),
        )

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
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @property
    def format(self) -> BassNotesFormat:
        return self._format

    @format.setter
    def format(self, format: BassNotesFormat):
        format = BassNotesFormat(format)
        if self.format == format:
            return
        if format == BassNotesFormat.INTERVAL:
            new_formatted_column = "bass_note_over_local_tonic"
        elif format == BassNotesFormat.FIFTHS:
            new_formatted_column = "bass_note"
        elif format == BassNotesFormat.SCALE_DEGREE:
            new_formatted_column = "bass_degree_and_mode"
        elif format == BassNotesFormat.SCALE_DEGREE_MAJOR:
            new_formatted_column = "bass_degree_major"
        elif format == BassNotesFormat.SCALE_DEGREE_MINOR:
            new_formatted_column = "bass_degree_minor"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def formatted_column(self) -> str:
        if self.format == BassNotesFormat.SCALE_DEGREE:
            if "mode" in self.default_groupby:
                return "bass_degree"
            else:
                return "bass_degree_and_mode"
        if self._formatted_column is not None:
            return self._formatted_column
        if self._default_formatted_column is not None:
            return self._default_formatted_column
        return

    def _modify_name(self):
        """Modify the :attr:`resource_name` to reflect the feature."""
        self.resource_name = f"{self.resource_name}.bass_notes"

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_bass_notes_feature(feature_df)
        feature_df = add_chord_tone_intervals(feature_df)
        feature_df = add_chord_tone_scale_degrees(feature_df)
        return self._sort_columns(feature_df)


def extend_cadence_feature(
    feature_df,
):
    columns_to_add = (
        "cadence_type",
        "cadence_subtype",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    if "cadence" not in feature_df.columns:
        raise DataframeIsMissingExpectedColumnsError(
            "cadence",
            feature_df.columns.to_list(),
        )
    split_labels = feature_df.cadence.str.split(".", expand=True).rename(
        columns={0: "cadence_type", 1: "cadence_subtype"}
    )
    feature_df = pd.concat([feature_df, split_labels], axis=1)
    return feature_df


class CadenceLabelFormat(FriendlyEnum):
    """Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the subtype, reducing more
    specific labels, whereas SUBTYPE displays subtypes only, omitting all labels that do not specify one.
    """

    RAW = "RAW"
    TYPE = "TYPE"
    SUBTYPE = "SUBTYPE"


class CadenceLabels(DcmlAnnotations):
    _auxiliary_column_names = ["label", "chord"]
    _convenience_column_names = (
        ["globalkey", "localkey"]
        + KEY_CONVENIENCE_COLUMNS
        + [
            "cadence_type",
            "cadence_subtype",
        ]
    )
    _feature_column_names = ["cadence"]
    _default_value_column = "cadence"
    _default_analyzer = "CadenceCounter"
    _extractable_features = None

    class Schema(DcmlAnnotations.Schema):
        format = FriendlyEnumField(
            CadenceLabelFormat,
            load_default=CadenceLabelFormat.RAW,
            metadata=dict(
                expose=True,
                description="Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the subtype, "
                "reducing more specific labels, whereas SUBTYPE displays subtypes only, omitting all "
                "labels that do not specify one.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = CadenceLabelFormat.RAW,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @property
    def format(self) -> CadenceLabelFormat:
        return self._format

    @format.setter
    def format(self, format: CadenceLabelFormat):
        format = CadenceLabelFormat(format)
        if self.format == format:
            return
        if format == CadenceLabelFormat.RAW:
            new_formatted_column = "cadence"
        elif format == CadenceLabelFormat.TYPE:
            new_formatted_column = "cadence_type"
        elif format == CadenceLabelFormat.SUBTYPE:
            new_formatted_column = "cadence_subtype"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        try:
            feature_df = extend_keys_feature(feature_df)
        except DataframeIsMissingExpectedColumnsError:
            pass
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_cadence_feature(feature_df)
        return self._sort_columns(feature_df)


class KeyAnnotations(DcmlAnnotations):
    _auxiliary_column_names = ["label"]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS
    _feature_column_names = ["globalkey", "localkey"]
    _extractable_features = None
    _default_value_column = "localkey_and_mode"

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = extend_keys_feature(feature_df)
        groupby_levels = feature_df.index.names[:-1]
        group_keys, _ = make_adjacency_groups(
            feature_df.localkey, groupby=groupby_levels
        )
        feature_df = condense_dataframe_by_groups(feature_df, group_keys)
        return self._sort_columns(feature_df)


# endregion Annotations
# region Controls


class Articulation(Feature):
    pass


# endregion Controls
# region Events


class NotesFormat(FriendlyEnum):
    """Format to display the notes in. NAME stands for note names, FIFTHS for the number of fifths from C,
    and MIDI for MIDI numbers."""

    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"


def merge_tied_notes(feature_df, groupby=None):
    expected_columns = ("duration", "tied", "midi", "staff")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    unique_values = feature_df.tied.unique()
    if 0 not in unique_values and -1 not in unique_values:
        # no tied notes (only <NA>) or has already been tied (only not-null value is 1)
        return feature_df
    if groupby is None:
        return merge_ties(feature_df)
    else:
        return feature_df.groupby(groupby, group_keys=False).apply(merge_ties)


def extend_notes_feature(feature_df):
    if "tpc_name" in feature_df.columns:
        return feature_df
    concatenate_this = [
        feature_df,
        ms3.transform(feature_df.tpc, ms3.tpc2name).rename("tpc_name"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class Notes(Feature):
    _auxiliary_column_names = [
        "chord_id",
        "gracenote",
        "midi",
        "name",
        "nominal_duration",
        "octave",
        "scalar",
        "tied",
        "tremolo",
    ]
    _convenience_column_names = [
        "tpc_name",
    ]
    _feature_column_names = ["tpc"]
    _default_analyzer = "PitchClassVectors"
    _default_value_column = "tpc"

    class Schema(Feature.Schema):
        format = FriendlyEnumField(
            NotesFormat,
            load_default=NotesFormat.NAME,
            metadata=dict(
                expose=True,
                description="Format to display the notes in. NAME stands for note names, FIFTHS for the number of "
                "fifths from C, and MIDI for MIDI numbers.",
            ),
        )
        merge_ties = mm.fields.Boolean(
            load_default=False,
            metadata=dict(
                title="Merge tied notes",
                expose=True,
                description="If set, notes that are tied together in the score are merged together, counting them "
                "as a single event of the corresponding length. Otherwise, every note head is counted.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                expose=True,
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = False,
        weight_grace_notes: float = 0.0,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        self._merge_ties = bool(merge_ties)
        self._weight_grace_notes = float(weight_grace_notes)

    @property
    def format(self) -> NotesFormat:
        return self._format

    @format.setter
    def format(self, format: NotesFormat):
        format = NotesFormat(format)
        if self.format == format:
            return
        if format == NotesFormat.NAME:
            new_formatted_column = "tpc_name"
        elif format == NotesFormat.FIFTHS:
            new_formatted_column = "tpc"
        elif format == NotesFormat.MIDI:
            new_formatted_column = "midi"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def merge_ties(self) -> bool:
        return self._merge_ties

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        if self.merge_ties:
            try:
                groupby = self.get_grouping_levels(UnitOfAnalysis.PIECE)
            except ResourceIsMissingPieceIndexError:
                groupby = None
                self.logger.info(
                    "Dataframe has no piece index. Merging ties without grouping."
                )
            feature_df = merge_tied_notes(feature_df, groupby=groupby)
        if self.weight_grace_notes:
            feature_df = ms3.add_weighted_grace_durations(
                feature_df, self.weight_grace_notes
            )
        feature_df = extend_notes_feature(feature_df)
        return self._sort_columns(feature_df)


# endregion Events
# region Structure
class Measures(Feature):
    pass


# endregion Structure
# region helpers


# endregion helpers
