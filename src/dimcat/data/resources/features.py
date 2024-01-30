from __future__ import annotations

import itertools
import logging
from typing import Callable, Hashable, Iterable, List, Literal, Optional

import frictionless as fl
import marshmallow as mm
import ms3
import numpy as np
import pandas as pd
from dimcat import DimcatConfig
from dimcat.base import FriendlyEnum, FriendlyEnumField
from dimcat.data.resources.base import D, FeatureName, S
from dimcat.data.resources.dc import (
    HARMONY_FEATURE_NAMES,
    DimcatIndex,
    Feature,
    Playthrough,
    SliceIntervals,
    UnitOfAnalysis,
)
from dimcat.data.resources.results import PhraseData, PhraseDataFormat
from dimcat.data.resources.utils import (
    get_corpus_display_name,
    join_df_on_index,
    merge_ties,
    safe_row_tuple,
)
from dimcat.dc_exceptions import (
    DataframeIsMissingExpectedColumnsError,
    FeatureIsMissingFormatColumnError,
    ResourceIsMissingPieceIndexError,
)
from dimcat.utils import get_middle_composition_year
from typing_extensions import Self

module_logger = logging.getLogger(__name__)


class Metadata(Feature):
    _default_analyzer = dict(dtype="Proportions", dimension_column="length_qb")
    _default_value_column = "piece"

    @property
    def metadata(self) -> Self:
        return self

    @metadata.setter
    def metadata(self, _):
        raise RuntimeError("Cannot set the property Metadata.metadata.")

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
        name: str = "mean_composition_year",
    ):
        group_cols = self._resolve_group_cols_arg(group_cols)
        years = get_middle_composition_year(metadata=self.df).rename(name)
        if not group_cols:
            return years
        result = years.groupby(group_cols).mean()
        return result

    def get_corpus_names(
        self,
        func: Callable[[str], str] = get_corpus_display_name,
    ):
        """Returns the corpus names in chronological order, based on their pieces' mean composition years.
        If ``func`` is specify, the function will be applied to each corpus name. This is useful for prettifying
        the names, e.g. by removing underscores.
        """
        mean_composition_years = self.get_composition_years(group_cols="corpus")
        sorted_corpus_names = mean_composition_years.sort_values().index.to_list()
        if func is None:
            return sorted_corpus_names
        return [func(corp) for corp in sorted_corpus_names]


# region Annotations

AUXILIARY_DCML_ANNOTATIONS_COLUMNS = [
    "label",
    "globalkey",
    "localkey",
    "pedal",
    "chord",
    "special",
    "numeral",
    "form",
    "figbass",
    "changes",
    "relativeroot",
    "cadence",
    "phraseend",
    "chord_type",
    "globalkey_is_minor",
    "localkey_is_minor",
    "chord_tones",
    "added_tones",
    "root",
    "bass_note",
    "alt_label",
    "pedalend",
    "placement",
    "color",
    "color_a",
    "color_b",
    "color_g",
    "color_r",
]
"""These columns are included in sub-features of HarmonyLabels to enable more means of investigation,
such as groupers."""

BASS_NOTE_CONVENIENCE_COLUMNS = [
    "bass_degree",
    "bass_degree_and_mode",
    "bass_degree_major",
    "bass_degree_minor",
    "bass_note_over_local_tonic",
]

CHORD_TONE_INTERVALS_COLUMNS = [
    "intervals_over_bass",
    "intervals_over_root",
]

CHORD_TONE_SCALE_DEGREES_COLUMNS = [
    "scale_degrees",
    "scale_degrees_and_mode",
    "scale_degrees_major",
    "scale_degrees_minor",
]

HARMONY_FEATURE_COLUMNS = [
    "root_roman",  # numeral/relativeroot
    "relativeroot_resolved",
    "effective_localkey",  # relativeroot/localkey (combined)
    "effective_localkey_resolved",  # relativeroot_resolved resolved against localkey
    "effective_localkey_is_minor",
    "pedal_resolved",
    "chord_and_mode",
    "chord_reduced",  # without parentheses ('changes')
    "chord_reduced_and_mode",
    "applied_to_numeral",  # if relativeroot is recursive, only the component following the last slash / (i.e. the
    # lowest level, which can be interpreted as the current localkey's numeral being elaborated)
    "numeral_or_applied_to_numeral",  # like the previous but missing values filled with 'numeral'
]


HARMONY_CONVENIENCE_COLUMNS = (
    HARMONY_FEATURE_COLUMNS
    + CHORD_TONE_INTERVALS_COLUMNS
    + CHORD_TONE_SCALE_DEGREES_COLUMNS
)
"""These columns are included in all :class:`Annotations` features that grant full access to DCML harmony labels.
First and foremost, this includes :class:`HarmonyLabels`, but also :class:`PhraseAnnotations` and derivatives.
"""

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


class Annotations(Feature):
    pass


class DcmlAnnotations(Annotations):
    _auxiliary_column_names = AUXILIARY_DCML_ANNOTATIONS_COLUMNS
    _convenience_column_names = None
    _feature_column_names = ["label"]
    _default_value_column = "label"
    _extractable_features = HARMONY_FEATURE_NAMES + (
        FeatureName.CadenceLabels,
        FeatureName.PhraseLabels,
    )

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Called by :meth:`_set_dataframe` to transform the dataframe before incorporating it.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        return self._sort_columns(feature_df)


class HarmonyLabelsFormat(FriendlyEnum):
    """Format to display the chord labels in. ROMAN stands for Roman numerals, ROMAN_REDUCED for the same numerals
    without any suspensions, alterations, additions, etc."""

    ROMAN = "ROMAN"
    ROMAN_REDUCED = "ROMAN_REDUCED"
    SCALE_DEGREE = "SCALE_DEGREE"
    SCALE_DEGREE_MAJOR = "SCALE_DEGREE_MAJOR"
    SCALE_DEGREE_MINOR = "SCALE_DEGREE_MINOR"


class HarmonyLabels(DcmlAnnotations):
    """A sub-feature of DcmlAnnotations which does not include any non-chord rows."""

    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + HARMONY_CONVENIENCE_COLUMNS
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
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        format: HarmonyLabelsFormat = HarmonyLabelsFormat.ROMAN,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            format:
                Format to display the chord labels in. ROMAN stands for Roman numerals,
                ROMAN_REDUCED for the same numerals without any suspensions, alterations, additions,
                etc.
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
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
    _convenience_column_names = (
        HarmonyLabels._convenience_column_names + BASS_NOTE_CONVENIENCE_COLUMNS
    )
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
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        format: NotesFormat = BassNotesFormat.INTERVAL,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            format=format,
            playthrough=playthrough,
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

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Called by :meth:`_set_dataframe` to transform the dataframe before incorporating it.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = extend_bass_notes_feature(feature_df)
        return self._sort_columns(feature_df)


class CadenceLabelFormat(FriendlyEnum):
    """Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the subtype, reducing more
    specific labels, whereas SUBTYPE displays subtypes only, omitting all labels that do not specify one.
    """

    RAW = "RAW"
    TYPE = "TYPE"
    SUBTYPE = "SUBTYPE"


class CadenceLabels(DcmlAnnotations):
    _auxiliary_column_names = ["label", "chord", "globalkey", "localkey"]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "cadence_type",
        "cadence_subtype",
    ]
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
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        format: NotesFormat = CadenceLabelFormat.RAW,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            format:
                Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the
                subtype, reducing more specific labels, whereas SUBTYPE displays subtypes only,
                omitting all labels that do not specify one.
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
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


class KeyAnnotations(DcmlAnnotations):
    _auxiliary_column_names = ["label"]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS
    _feature_column_names = ["globalkey", "localkey"]
    _extractable_features = None
    _default_value_column = "localkey_and_mode"


def make_sequence_non_repeating(
    sequence: S,
) -> tuple:
    """Returns values in the given sequence without immediate repetitions. Fails if the sequence contains NA."""
    return tuple(val for val, _ in itertools.groupby(sequence))


def _condense_component(
    component_df: D,
    qstamp_col_position: int,
    duration_col_position: int,
    localkey_col_position: int,
    label_col_position: int,
    chord_col_position: int,
) -> S:
    """Returns a series which condenses the phrase components into a row."""
    first_row = component_df.iloc[0]
    component_info = _compile_component_info(
        component_df,
        qstamp_col_position,
        duration_col_position,
        localkey_col_position,
        label_col_position,
        chord_col_position,
    )
    row_values = first_row.to_dict()
    row_values.update(component_info)
    return pd.Series(row_values, name=first_row.name)


def _compile_component_info(
    component_df: D,
    qstamp_col_position,
    duration_col_position,
    localkey_col_position,
    label_col_position,
    chord_col_position,
    key_prefix: Optional[str] = "",
):
    start_qstamp = component_df.iat[0, qstamp_col_position]
    end_qstamp = (
        component_df.iat[-1, qstamp_col_position]
        + component_df.iat[-1, duration_col_position]
    )
    new_duration = float(end_qstamp - start_qstamp)
    columns = component_df.iloc(axis=1)
    localkeys = tuple(columns[localkey_col_position])
    modulations = make_sequence_non_repeating(localkeys)
    labels = tuple(columns[label_col_position])
    chords = tuple(columns[chord_col_position])
    component_info = dict(
        localkeys=localkeys,
        n_modulations=len(modulations) - 1,
        modulatory_sequence=modulations,
        n_labels=len(labels),
        labels=labels,
        n_chords=len(chords),
        chords=chords,
    )
    duration_key = "duration_qb"
    if key_prefix:
        component_info = {
            f"{key_prefix}{key}": val for key, val in component_info.items()
        }
        if key_prefix != "phrase_":
            # phrase duration is used as the main 'duration_qb' column
            duration_key = f"{key_prefix}duration_qb"
    component_info[duration_key] = new_duration
    return component_info


def condense_components(raw_phrase_df: D) -> D:
    qstamp_col_position = raw_phrase_df.columns.get_loc("quarterbeats")
    duration_col_position = raw_phrase_df.columns.get_loc("duration_qb")
    localkey_col_position = raw_phrase_df.columns.get_loc("localkey")
    label_col_position = raw_phrase_df.columns.get_loc("label")
    chord_col_position = raw_phrase_df.columns.get_loc("chord")
    groupby_levels = raw_phrase_df.index.names[:-1]
    return raw_phrase_df.groupby(groupby_levels).apply(
        _condense_component,
        qstamp_col_position,
        duration_col_position,
        localkey_col_position,
        label_col_position,
        chord_col_position,
    )


def _condense_phrase(
    phrase_df: D,
    qstamp_col_position: int,
    duration_col_position: int,
    localkey_col_position: int,
    label_col_position: int,
    chord_col_position: int,
) -> dict:
    """Returns a series which condenses the phrase into a row."""
    component_indices = phrase_df.groupby("phrase_component").indices
    body_idx = component_indices.get("body")
    codetta_idx = component_indices.get("codetta")
    first_phrase_i = body_idx[0]
    last_body_i = body_idx[-1]
    end_label = phrase_df.iat[last_body_i, label_col_position]
    end_chord = phrase_df.iat[last_body_i, chord_col_position]
    if "}" in end_label:
        interlocked_ante = "}" in phrase_df.iat[first_phrase_i, label_col_position]
        interlocked_post = codetta_idx is None
    else:
        # old-style phrase endings didn't provide the means to encode phrase interlocking
        interlocked_ante, interlocked_post = pd.NA, pd.NA
    if codetta_idx is None:
        # if no codetta is defined, the phrase info will simply be copied from the body component
        component_index_iterable = component_indices.items()
    else:
        phrase_idx = np.concatenate([body_idx[:-1], codetta_idx])
        component_index_iterable = [("phrase", phrase_idx), *component_indices.items()]
    first_body_row = phrase_df.iloc[first_phrase_i]
    row_values = first_body_row.to_dict()
    for group, component_df in (
        (group, phrase_df.take(idx)) for group, idx in component_index_iterable
    ):
        component_info = _compile_component_info(
            component_df,
            qstamp_col_position,
            duration_col_position,
            localkey_col_position,
            label_col_position,
            chord_col_position,
            key_prefix=f"{group}_",
        )
        row_values.update(component_info)
    if codetta_idx is None:
        phrase_info = {}
        for key, value in component_info.items():
            if key.startswith("body_"):
                if key == "body_duration_qb":
                    phrase_key = "duration_qb"
                else:
                    phrase_key = key.replace("body_", "phrase_")
                phrase_info[phrase_key] = value
        row_values.update(phrase_info)
    row_values["interlocked_ante"] = interlocked_ante
    row_values["interlocked_post"] = interlocked_post
    row_values["end_label"] = end_label
    row_values["end_chord"] = end_chord
    return row_values


def condense_phrases(raw_phrase_df: D) -> D:
    qstamp_col_position = raw_phrase_df.columns.get_loc("quarterbeats")
    duration_col_position = raw_phrase_df.columns.get_loc("duration_qb")
    localkey_col_position = raw_phrase_df.columns.get_loc("localkey")
    label_col_position = raw_phrase_df.columns.get_loc("label")
    chord_col_position = raw_phrase_df.columns.get_loc("chord")
    # we're not using :meth:`pandas.DataFrameGroupBy.apply` because the series returned by _condense_phrases may have
    # varying lengths, which would result in a series, not a dataframe. Instead, we're collecting groupwise row dicts
    # and then creating a dataframe from them.
    groupby_levels = raw_phrase_df.index.names[:-2]
    group2dict = {
        group: _condense_phrase(
            phrase_df,
            qstamp_col_position,
            duration_col_position,
            localkey_col_position,
            label_col_position,
            chord_col_position,
        )
        for group, phrase_df in raw_phrase_df.groupby(groupby_levels)
    }
    result = pd.DataFrame.from_dict(group2dict, orient="index")
    result.index.names = groupby_levels
    nullable_int_cols = {
        col_name: "Int64"
        for comp, col in itertools.product(
            ("phrase_", "ante_", "body_", "codetta_", "post_"),
            ("n_modulations", "n_labels", "n_chords"),
        )
        if (col_name := comp + col) in result.columns
    }
    result = result.astype(nullable_int_cols)
    return result


def tuple_contains(series_with_tuples: S, *values: Hashable):
    """Function that can be used in queries passed to :meth:`PhraseLabels.filter_phrase_data` to select rows in which
    the column's tuples contain any of the given values.

    Example

    """
    values = set(values)
    return series_with_tuples.map(values.intersection).astype(bool)


class PhraseAnnotations(HarmonyLabels):
    _extractable_features = [FeatureName.PhraseComponents, FeatureName.PhraseLabels]

    class Schema(DcmlAnnotations.Schema):
        n_ante = mm.fields.Int(
            metadata=dict(
                expose=True,
                description="Specify an integer > 0 in order to include additional information on the n labels "
                "preceding the phrase. These are generally part of a previous phrase.",
            )
        )
        n_post = mm.fields.Int(
            metadata=dict(
                expose=True,
                description="Specify an integer > 0 in order to include additional information on the n labels "
                "following the phrase. These are generally part of a subsequent phrase.",
            )
        )

    def __init__(
        self,
        n_ante: int = 0,
        n_post: int = 0,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        format=None,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            n_ante:
                By default, each phrase includes information about the included labels from beginning to end. Specify an
                integer > 0 in order to include additional information on the n labels preceding the phrase. These
                are generally part of a previous phrase.
            n_post:
                By default, each phrase includes information about the included labels from beginning to end. Specify an
                integer > 0 in order to include additional information on the n labels following the phrase. These
                are generally part of a subsequent phrase.
            format: Not in use.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            format=format,
            playthrough=playthrough,
        )
        self.n_ante = n_ante
        self.n_post = n_post

    @property
    def phrase_df(self) -> D:
        """Alias for :meth:`df`."""
        return self.df

    def get_phrase_data(
        self,
        columns: str | List[str] = "label",
        components: PhraseComponentName
        | Literal["phrase"]
        | Iterable[PhraseComponentName] = "body",
        query: Optional[str] = None,
        reverse: bool = False,
        level_name: str = "i",
        wide_format: bool = False,
        drop_levels: bool | int | str | Iterable[int | str] = False,
        drop_duplicated_ultima_rows: Optional[bool] = None,
    ) -> PhraseData:
        """

        Args:
            columns:
                Column(s) to include in the result.
            components:
                Which of the four phrase components to include, ∈ {'ante', 'body', 'codetta', 'post'}.
                For convenience, the string 'phrase' is also accepted, which is equivalent to ["body", "codetta"] and
                ``drop_duplicated_ultima_rows=True``.
            query:
                A convenient way to include only those phrases in the result that match the criteria
                formulated in the string query. A query is a string and generally takes the form
                "<column_name> <operator> <value>". Several criteria can be combined using boolean
                operators, e.g. "localkey_mode == 'major' & label.str.contains('/')". This option
                is particularly interesting when used on :class:`PhraseLabels` because it enables
                queries based on the properties of phrases such as
                "body_n_modulations == 0 & end_label.str.contains('IAC')".  For the columns
                containing tuples, you can used a special function to filter those rows that
                contain any of the specified values:
                "@tuple_contains(body_chords, 'V(94)', 'V(9)', 'V(4)')".
            reverse:
                Pass True to reverse the order of harmonies so that each phrase's last label comes
                first.
            level_name:
                Defaults to 'i', which is the name of the original level that will be replaced
                by this new one. The new one represents the individual integer range for each
                phrase, starting at 0.
            wide_format:
                Pass True to unstack the result so that the columns for each phrase are concatenated
                side by side.
            drop_levels:
                Can be a boolean or any level specifier accepted by :meth:`pandas.MultiIndex.droplevel()`.
                If False (default), all levels are retained. If True, only the phrase_id level and
                the ``level_name`` are retained. In all other cases, the indicated (string or
                integer) value(s) must be valid and cause one of the index levels to be dropped.
                ``level_name`` cannot be dropped. Dropping 'phrase_id' will likely lead to an
                exception if a :class:`PhraseData` object will be displayed in WIDE format.
            drop_duplicated_ultima_rows:
                The default behaviour (when None), depends on the value of ``components``: If you set
                ``components='phrase'``, this setting defaults to True, otherwise to False; where
                False corresponds to the default where  each phrase body ends on a duplicate of the
                phrase's ultima label, with zero-duration, enabling the creation of PhraseData
                containing only phrase bodies (i.e., ``components='body'``), without losing information
                about the ultima label. When analyzing entire phrases, however, these duplicate
                rows may be unwanted and can be dropped by setting this option to True.

        Returns:
            Dataframe representing partial information on the selected phrases in long or wide format.
        """
        df_format = PhraseDataFormat.WIDE if wide_format else PhraseDataFormat.LONG
        analyzer = dict(
            dtype="PhraseDataAnalyzer",
            columns=columns,
            components=components,
            query=query,
            reverse=reverse,
            level_name=level_name,
            format=df_format,
            drop_levels=drop_levels,
            drop_duplicated_ultima_rows=drop_duplicated_ultima_rows,
        )
        return self.apply_step(analyzer)

    def _prepare_feature_df(self, feature_config: DimcatConfig) -> D:
        """Called by :meth:`_extract_feature`, returns the raw PhraseAnnotations dataframe."""
        return self.phrase_df

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Called by :meth:`_set_dataframe` to transform the dataframe before incorporating it.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        self._phrase_df = feature_df
        return feature_df


class PhraseComponentName(FriendlyEnum):
    ANTE = "ante"
    BODY = "body"
    CODETTA = "codetta"
    POST = "post"


class PhraseComponents(PhraseAnnotations):
    _convenience_column_names = HarmonyLabels._convenience_column_names + [
        "localkeys",
        "n_modulations",
        "modulatory_sequence",
        "n_labels",
        "labels",
        "n_chords",
        "chords",
    ]
    _default_value_column = "chords"
    _feature_column_names = ["chords"]

    @property
    def phrase_df(self) -> D:
        """Returns the df that corresponds to the :class:`PhraseAnnotations` feature from which the
        PhraseComponents were derived.
        """
        return self._phrase_df

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Condense the raw PhraseAnnotations dataframe into a dataframe with one row per phrase component."""
        feature_df = super()._adapt_newly_set_df(feature_df)
        return condense_components(feature_df)


class PhraseLabels(PhraseAnnotations):
    _convenience_column_names = HarmonyLabels._convenience_column_names + [
        "phrase_localkeys",
        "phrase_n_modulations",
        "phrase_modulatory_sequence",
        "phrase_n_labels",
        "phrase_labels",
        "phrase_n_chords",
        "phrase_chords",
        "body_localkeys",
        "body_n_modulations",
        "body_modulatory_sequence",
        "body_n_labels",
        "body_labels",
        "body_n_chords",
        "body_chords",
        "body_duration_qb",
        "codetta_localkeys",
        "codetta_n_modulations",
        "codetta_modulatory_sequence",
        "codetta_n_labels",
        "codetta_labels",
        "codetta_n_chords",
        "codetta_chords",
        "codetta_duration_qb",
        "interlocked_ante",
        "interlocked_post",
        "end_label",
        "end_chord",
    ]
    _default_value_column = "phrase_chords"
    _feature_column_names = ["phrase_chords"]

    @property
    def phrase_df(self) -> D:
        """Returns the df that corresponds to the :class:`PhraseAnnotations` feature from which the
        PhraseLabels were derived.
        """
        return self._phrase_df

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Condense the raw PhraseAnnotations dataframe into a dataframe with one row per phrase."""
        feature_df = super()._adapt_newly_set_df(feature_df)
        return condense_phrases(feature_df)


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
                description="If False (default), each row corresponds to a note head, even if it does not the full "
                "duration of the represented sounding event or even an onset. Setting to True results in "
                "notes being tied over to from a previous note to be merged into a single note with the "
                "summed duration. After the transformation, only note heads that actually represent a note "
                "onset remain.",
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
        merge_ties: bool = False,
        weight_grace_notes: float = 0.0,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        format: NotesFormat = NotesFormat.NAME,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            merge_ties:
                If False (default), each row corresponds to a note head, even if it does not the full duration of the
                represented sounding event or even an onset. Setting to True results in notes being tied over to from a
                previous note to be merged into a single note with the summed duration. After the transformation,
                only note heads that actually represent a note onset remain.
            weight_grace_notes:
                Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have duration 0
                and are therefore excluded from many statistics.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            format:
                :attr:`format`. Format to display the notes in. The default NAME stands for note names, FIFTHS for
                the number of fifths from C, and MIDI for MIDI numbers.
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
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

    def _adapt_newly_set_df(self, feature_df: D) -> D:
        """Called by :meth:`_set_dataframe` to transform the dataframe before incorporating it.
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
