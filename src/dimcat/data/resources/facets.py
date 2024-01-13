from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple  # , ClassVar, Tuple

import frictionless as fl
import ms3
import numpy as np
import pandas as pd
from dimcat.base import DimcatConfig, ObjectEnum
from dimcat.data.resources import DimcatResource, FeatureName, Metadata, Resource
from dimcat.data.resources.base import D, S
from dimcat.data.resources.dc import HARMONY_FEATURE_NAMES, Playthrough
from dimcat.data.resources.features import (
    CHORD_TONE_INTERVALS_COLUMNS,
    CHORD_TONE_SCALE_DEGREES_COLUMNS,
    HARMONY_FEATURE_COLUMNS,
    CadenceLabels,
    DcmlAnnotations,
    KeyAnnotations,
    PhraseAnnotations,
)
from dimcat.data.resources.utils import (
    apply_playthrough,
    boolean_is_minor_column_to_mode,
    condense_dataframe_by_groups,
    condense_pedal_points,
    drop_rows_with_missing_values,
    make_adjacency_groups,
    make_group_start_mask,
    make_groups_lasts_mask,
    safe_row_tuple,
    tuple2str,
    update_duration_qb,
)
from dimcat.dc_exceptions import DataframeIsMissingExpectedColumnsError
from numpy import typing as npt
from typing_extensions import Self

module_logger = logging.getLogger(__name__)

# region helpers


def add_chord_tone_scale_degrees(
    feature_df,
):
    """Turns 'chord_tones' column into multiple scale-degree columns."""
    columns_to_add = CHORD_TONE_SCALE_DEGREES_COLUMNS
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
    columns_to_add = CHORD_TONE_INTERVALS_COLUMNS
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


def extend_keys_feature(
    feature_df,
):
    columns_to_add = (
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",  # resolves relative keys such as V/V (to II)
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
            feature_df, ms3.resolve_relative_keys, ["localkey", "globalkey_is_minor"]
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
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = HARMONY_FEATURE_COLUMNS
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = (
        "chord",
        "form",
        "figbass",
        "pedal",
        "numeral",
        "relativeroot",
        "globalkey_is_minor",
        "localkey_is_minor",
        "localkey_mode",
        "localkey_resolved",
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
    if "relativeroot_resolved" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df,
                ms3.resolve_relative_keys,
                ["relativeroot", "localkey_is_minor"],
            ).rename("relativeroot_resolved")
        )
    if "effective_localkey" not in feature_df.columns:
        concatenate_this.append(
            (
                effective_localkey := (
                    (feature_df.relativeroot + "/").fillna("")
                    + feature_df.localkey_resolved
                ).rename("effective_localkey")
            )
        )
        effective_localkey_and_mode = pd.concat(
            [effective_localkey, feature_df.globalkey_is_minor], axis=1
        )
        concatenate_this.append(
            (
                effective_localkey_resolved := ms3.transform(
                    effective_localkey_and_mode, ms3.resolve_relative_keys
                ).rename("effective_localkey_resolved")
            )
        )
    else:
        effective_localkey_resolved = feature_df.effective_localkey_resolved
    if "effective_localkey_is_minor" not in feature_df.columns:
        concatenate_this.append(
            effective_localkey_resolved.str.islower()
            .fillna(feature_df.localkey_is_minor)
            .rename("effective_localkey_is_minor")
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
    if "applied_to_numeral" not in feature_df.columns:
        applied_to_numeral = feature_df.relativeroot.str.split("/").map(
            lambda lst: lst[-1], na_action="ignore"
        )
        concatenate_this.append(applied_to_numeral.rename("applied_to_numeral"))
    else:
        applied_to_numeral = feature_df.applied_to_numeral
    if "numeral_or_applied_to_numeral" not in feature_df.columns:
        concatenate_this.append(
            applied_to_numeral.copy()
            .fillna(feature_df.numeral)
            .rename("numeral_or_applied_to_numeral")
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


def _get_body_end_positions_from_raw_phrases(phrase_df: D) -> List[int]:
    """Returns for each phrase body the index position of the last row. Typical input is a dataframe representing a
    MultiIndex. Expects the columns 'phrase_id' and  'phrase_component'. If the latter is present, all components
    except 'body' are disregarded. If not, phrase sequences are expected to be bodies only.
    """
    body_end_positions = []
    if "phrase_component" in phrase_df.columns:
        for (phrase_id, phrase_component), idx in phrase_df.groupby(
            ["phrase_id", "phrase_component"]
        ).indices.items():
            if phrase_component != "body":
                continue
            body_end_positions.append(idx[-1])
    else:
        for idx in phrase_df.groupby("phrase_id").indices.values():
            body_end_positions.append(idx[-1])
    return body_end_positions


def _get_body_start_positions_from_raw_phrases(phrase_df: D) -> List[int]:
    """Returns for each phrase body the index position of the first row."""
    body_start_positions = []
    phrase_df = phrase_df.reset_index()
    if "phrase_component" in phrase_df.columns:
        for (phrase_id, phrase_component), idx in phrase_df.groupby(
            ["phrase_id", "phrase_component"]
        ).indices.items():
            if phrase_component != "body":
                continue
            body_start_positions.append(idx[0])
    else:
        for phrase_id, idx in phrase_df.groupby("phrase_id").indices.items():
            body_start_positions.append(idx[0])
    return body_start_positions


def _get_index_intervals_for_phrases(
    markers: S,
    n_ante: int = 0,
    n_post: int = 0,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[int, int, Optional[int], int, int]]:
    """Expects a Series with a RangeIndex and computes (from, to) index position intervals based on the presence of
    either the start_symbol or the end_symbol. If both are found, an error is thrown. If None is found, the result is
    an empty list.

    The function operates based on the constants

        start_symbol ``"{"``
            If this symbol is present in any of the series' strings, intervals will be formed starting from one to the
            next occurrences (within strings). The interval for the last symbol reaches until the end of the series
            (that is, the last index position + 1).

        end_symbol ``"\\"``
            If this symbol is present in any of the series' strings, intervals will be formed starting from the first
            index position to the position of the first end_symbol + 1, and from there until one after the next, and
            so on.

    Args:
        markers:
            A Series containing either start or end symbols of phrases. Expected to have a RangeIndex. When the series
            corresponds to a chunk of a larger one, the RangeIndex should correspond to the respective positions in
            the original series.
        n_ante: Pass a positive integer to have the intervals include n earlier positions.
        n_post:
            Pass a positive integer > 0 to have the intervals include n subsequent positions. The minimum is 1 because
            for new-style phrase endings (``}``) the end_symbol may actually appear only with the beginning of the
            subsequent phrase in the case of ``}{``.
        logger:

    Returns:
        A list of (first_i, start_i, end_i, subsequent_i, stop_i) index positions that can be used for slicing rows
        of the dataframe from which the series was taken. The meaning of the included slice intervals is as follows:

        * ``[start_i:start_i)``: The n_ante positions before the phrase.
        * ``[start_i:end_i]``: The body of the phrase, including end symbol.
        * ``[end_i:subsequent_i)``:
          The codetta, i.e., the part between the end_symbol and the subsequent phrase. In the case of phrase overlap,
          the two are identical and the codetta is empty.
        * ``[subsequent_i:stop_i)``: The n_post positions after the phrase.
    """
    if logger is None:
        logger = module_logger
    present_symbols = markers.unique()
    start_symbol, end_symbol = "{", r"\\"
    has_start = start_symbol in present_symbols
    has_end = end_symbol in present_symbols
    if not (has_start or has_end):
        return []
    if has_start and has_end:
        logger.warning(
            f"Currently I can create phrases either based on end symbols or on start symbols, but this df has both:"
            f":\n{markers.value_counts().to_dict()}\nUsing {start_symbol}, ignoring {end_symbol}..."
        )
    ix_min = markers.index.min()
    ix_max = markers.index.max() + 1
    if has_start:
        end_symbol = "}"
        start_symbol_mask = markers.str.contains(start_symbol).fillna(False)
        starts_ix = start_symbol_mask.index[start_symbol_mask].to_list()
        end_symbol_mask = markers.str.contains(end_symbol).fillna(False)
        ends_ix = end_symbol_mask.index[end_symbol_mask].to_list()

        def include_end_ix(fro, to):
            potential = range(fro + 1, to + 1)
            included_ends = [ix for ix in ends_ix if ix in potential]
            n_ends = len(included_ends)
            if not n_ends:
                inspect_series = markers.loc[fro + 1 : to + 1].dropna()
                logger.warning(
                    f"Phrase [{fro}:{to}] was expected to have an end symbol within [{fro+1}:{to+1}]:\n{inspect_series}"
                )
                return (fro, None, to)
            elif n_ends > 2:
                inspect_series = markers.loc[fro + 1 : to + 1].dropna()
                logger.warning(
                    f"Phrase [{fro}:{to}] has multiple end symbols within [{fro+1}:{to+1}]:\n{inspect_series}"
                )
                return (fro, None, to)
            end_ix = included_ends[0]
            return (fro, end_ix, to)

        start_end_subsequent = [
            include_end_ix(fro, to)
            for fro, to in zip(starts_ix, starts_ix[1:] + [ix_max])
        ]
    else:
        end_symbol_mask = markers.str.contains(end_symbol).fillna(False)
        subsequent_ix = (end_symbol_mask.index[end_symbol_mask] + 1).to_list()
        start_end_subsequent = [
            (fro, to - 1, to)
            for fro, to in zip([ix_min] + subsequent_ix[:-1], subsequent_ix)
        ]
    result = []
    for start_i, end_i, subsequent_i in start_end_subsequent:
        first_i = start_i
        if n_ante:
            new_first_i = start_i - n_ante
            if new_first_i >= ix_min:
                first_i = new_first_i
            else:
                first_i = ix_min
        stop_i = subsequent_i
        if n_post:
            new_stop_i = subsequent_i + n_post
            if new_stop_i <= ix_max:
                stop_i = new_stop_i
            else:
                stop_i = ix_max
        result.append((first_i, start_i, end_i, subsequent_i, stop_i))
    return result


def get_index_intervals_for_phrases(
    harmony_labels: D,
    group_cols: List[str],
    n_ante: int = 0,
    n_post: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[Any, List[Tuple[int, int]]]:
    """Returns a list of slice intervals for selecting the rows belonging to a phrase."""
    if logger is None:
        logger = module_logger
    phraseends_reset = harmony_labels.reset_index()
    group_intervals = {}
    groupby = phraseends_reset.groupby(group_cols)
    for group, markers in groupby.phraseend:
        first_start_end_sbsq_last = _get_index_intervals_for_phrases(
            markers, n_ante=n_ante, n_post=n_post, logger=logger
        )
        group_intervals[group] = first_start_end_sbsq_last
    return group_intervals


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


def _make_concatenated_ranges(
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
):
    """Helper function that is a vectorized version of the equivalent but roughly 100x slower

    .. code-block:: python

       np.array([np.arange(start, stop) for start, stop in zip(starts, stops)]).flatten()

    Solution adapted from Warren Weckesser's via https://stackoverflow.com/a/20033438


    Args:
        starts: Array of index range starts.
        stops:  Array of index range stops (exclusive).
        counts: Corresponds to stops - starts. 0-count ranges need to be excluded beforehand.

    Returns:

    """

    counts1 = counts[:-1]
    reset_index = np.cumsum(counts1)
    reset_values = 1 + starts[1:] - stops[:-1]
    incr = np.ones(counts.sum(), dtype=int)
    incr[0] = starts[0]
    incr[reset_index] = reset_values
    incr.cumsum(out=incr)
    return incr


def _make_range_boundaries(
    first: int, start: int, end: int, sbsq: int, stop: int
) -> npt.NDArray[np.int64]:
    """Turns the individual tuples output by :func:`_get_index_intervals_for_phrases` into four range boundaries.
    The four intervals are [first:start), [start:end], [end:sbsq), [sbsq:stop), which correspond to the components
    (ante, body, codetta, post) of the phrase. The body interval is right-inclusive, which means that the end
    symbol is included both in the body and, in the beginning of the 'codetta' or 'post' component.
    """
    return np.array([[first, start], [start, end + 1], [end, sbsq], [sbsq, stop]])


def make_raw_phrase_df(
    feature_df: D,
    ix_intervals: List[Tuple[int, int, Optional[int], int, int]],
    logger: Optional[logging.Logger] = None,
):
    """Takes the intervals generated by :meth:`get_index_intervals_for_phrases` and returns a dataframe with two
    additional index levels, one expressing a running count of phrases used as IDs, and one exhibiting for each phrase
    between one and for of the phrase_component names (ante, body, codetta, post), where 'body' is guaranteed to be
    ,present.
    """
    if logger is None:
        logger = module_logger
    take_mask, id_level, name_level = make_take_mask_and_index(
        ix_intervals, logger=logger
    )
    phrase_df = feature_df.take(take_mask)
    old_index = phrase_df.index.to_frame(index=False)
    new_levels = pd.DataFrame(
        dict(
            phrase_id=id_level,
            phrase_component=name_level,
        )
    )
    nlevels = phrase_df.index.nlevels
    new_index = pd.concat(
        [
            old_index.take(range(nlevels - 1), axis=1),
            new_levels,
            old_index.take([-1], axis=1),
        ],
        axis=1,
    )
    phrase_df.index = pd.MultiIndex.from_frame(new_index)
    # here we correct durations for the fact that the end symbol is included both as last symbol of the body and the
    # first symbol of the codetta or subsequent phrase. At the end of the body, the duration is set to 0.
    body_end_positions = _get_body_end_positions_from_raw_phrases(new_index)
    duration_col_position = phrase_df.columns.get_loc("duration_qb")
    phrase_df.iloc[body_end_positions, duration_col_position] = 0.0
    components_lasts = make_groups_lasts_mask(
        new_index, ["phrase_id", "phrase_component"]
    )
    # ToDo: add to documentation the fact that the duration of terminal harmonies is not amended. This allows for
    # inspecting the duration of the last harmony but leads to the fact that the summed duration of all phrases in a
    # piece may be longer than the piece itself, namely when a long terminal harmony is 'interrupted' by the beginning
    # of the next phrase: the following code duplicates the duration following the {
    update_duration_qb(
        phrase_df, ~components_lasts, logger
    )  # ToDo: check 0-durations in codetta, e.g. for } labels; overhaul phrase duration update (condense_pedal_points)
    return phrase_df


def make_take_mask_and_index(
    ix_intervals: List[Tuple[int, int, Optional[int], int, int]],
    logger: logging.Logger,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Takes a list of (first_i, start_i, end_i, subsequent_i, stop_i) index positions and turns them into

    * an array of corresponding index positions that can be used as argument for :meth:`pandas.DataFrame.take`
    * an array of equal length that specifies the corresponding phrase IDs (which come from an integer range)
    * an array of equal length that specifies the corresponding phrase components (ante, body, codetta, post)
    """
    range_boundaries = []
    for first, start, end, sbsq, last in ix_intervals:
        if end is None:
            logger.info("Skipping phrase with undefined end symbol.")
            continue
        range_boundaries.append(_make_range_boundaries(first, start, end, sbsq, last))
    ranges = np.vstack(range_boundaries)
    starts, stops = ranges.T
    counts = stops - starts
    not_empty_mask = counts > 0
    if not_empty_mask.any():
        take_mask = _make_concatenated_ranges(
            starts[not_empty_mask], stops[not_empty_mask], counts[not_empty_mask]
        )
    else:
        take_mask = _make_concatenated_ranges(starts, stops, counts)
    n_repeats = int(counts.shape[0] / 4)
    phrase_ids = np.repeat(np.arange(n_repeats), 4)
    names = np.tile(np.array(["ante", "body", "codetta", "post"]), n_repeats)
    id_level = phrase_ids.repeat(counts)
    name_level = names.repeat(counts)
    return take_mask, id_level, name_level


# endregion helpers


class MuseScoreFacetName(ObjectEnum):
    MuseScoreChords = "MuseScoreChords"
    MuseScoreFacet = "MuseScoreFacet"
    MuseScoreHarmonies = "MuseScoreHarmonies"
    MuseScoreMeasures = "MuseScoreMeasures"
    MuseScoreNotes = "MuseScoreNotes"


class Facet(DimcatResource):
    """A facet is one aspect of a score that can sensibly ordered and conceived of along the score's timeline. The
    format of a facet depends on the score format and tries to stay as close to the original as possible, using only
    the necessary minimum of standardization. Content an format of a facet define which features can be extracted,
    based on which configuration options.
    """

    pass


class EventsFacet(Facet):
    pass


class ControlsFacet(Facet):
    pass


class AnnotationsFacet(Facet):
    pass


class StructureFacet(Facet):
    pass


class MuseScoreFacet(Facet):
    """A single facet of a MuseScore package as created by the ms3 MuseScore parsing library. Contains a single TSV
    facet one or several corpora. Naming format ``<name>.<facet>[.tsv]``."""

    _enum_type = MuseScoreFacetName

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        if isinstance(descriptor, (str, Path)):
            raise TypeError(
                f"This method expects a descriptor dictionary. In order to create a "
                f"{cls.name} from a path, use {cls.__name__}.from_descriptor_path() instead."
            )
        if cls.name == "MuseScoreFacet":
            # dispatch to the respective facet based on the resource name
            if isinstance(descriptor, fl.Resource):
                fl_resource = descriptor
            else:
                fl_resource = fl.Resource.from_descriptor(descriptor)
            facet_name2constructor = dict(
                chords=MuseScoreChords,
                expanded=MuseScoreHarmonies,
                harmonies=MuseScoreHarmonies,
                measures=MuseScoreMeasures,
                metadata=Metadata,
                notes=MuseScoreNotes,
            )
            resource_name = fl_resource.name
            try:
                _, facet_name = resource_name.rsplit(".", 1)
                Klass = facet_name2constructor.get(facet_name)
                if Klass is None:
                    raise NotImplementedError(
                        f"MuseScoreFacet {facet_name} is not implemented."
                    )
            except ValueError:
                if any(
                    resource_name.endswith(f_name) for f_name in facet_name2constructor
                ):
                    Klass = next(
                        klass
                        for f_name, klass in facet_name2constructor.items()
                        if resource_name.endswith(f_name)
                    )
            return Klass.from_descriptor(
                descriptor=descriptor,
                descriptor_filename=descriptor_filename,
                basepath=basepath,
                auto_validate=auto_validate,
                default_groupby=default_groupby,
            )
        return super().from_descriptor(
            descriptor=descriptor,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )


class MuseScoreChords(MuseScoreFacet, ControlsFacet):
    _extractable_features = (FeatureName.Articulation,)


class MuseScoreHarmonies(MuseScoreFacet, AnnotationsFacet):
    _extractable_features = (
        FeatureName.DcmlAnnotations,
        FeatureName.CadenceLabels,
        FeatureName.PhraseAnnotations,
        FeatureName.PhraseComponents,
        FeatureName.PhraseLabels,
    ) + HARMONY_FEATURE_NAMES

    def _prepare_feature_df(self, feature_config: DimcatConfig) -> D:
        Constructor = feature_config.options_class
        columns_to_load_if_available = Constructor.get_default_column_names()
        feature_df = self.get_dataframe(usecols=tuple(columns_to_load_if_available))
        return feature_df

    def _transform_df_for_extraction(
        self, feature_df: D, feature_config: DimcatConfig
    ) -> D:
        feature_name = FeatureName(feature_config.options_dtype)
        feature_settings = dict(feature_config.complete())
        if playthrough_value := feature_settings.get("playthrough"):
            playthrough = Playthrough(playthrough_value)
            if playthrough != Playthrough.RAW:
                feature_df = apply_playthrough(
                    feature_df, playthrough, logger=self.logger
                )
        cls = feature_name.get_class()
        feature_column_names = cls._feature_column_names
        if issubclass(cls, DcmlAnnotations):
            feature_df = extend_keys_feature(feature_df)
            if issubclass(cls, CadenceLabels):
                feature_df = drop_rows_with_missing_values(
                    feature_df, feature_column_names, logger=self.logger
                )
                feature_df = extend_cadence_feature(feature_df)
            elif issubclass(cls, KeyAnnotations):
                groupby_levels = feature_df.index.names[:-1]
                group_keys, _ = make_adjacency_groups(
                    feature_df.localkey, groupby=groupby_levels
                )
                feature_df = condense_dataframe_by_groups(
                    feature_df, group_keys, logger=self.logger
                )
            else:  # issubclass(cls, (HarmonyLabels, PhraseAnnotations))
                if issubclass(cls, PhraseAnnotations):
                    missing_mask = feature_df.chord.isna()
                    groupby_levels = feature_df.index.names[:-1]
                    group_start_mask = make_group_start_mask(feature_df, groupby_levels)
                    feature_df.loc[missing_mask, ["chord_tones", "added_tones"]] = pd.NA
                    ffill_mask = missing_mask | (
                        missing_mask.shift(-1).fillna(False) & ~group_start_mask
                    )
                    harmony_fill_columns = [
                        "pedal",
                        "chord",
                        "special",
                        "numeral",
                        "form",
                        "figbass",
                        "changes",
                        "relativeroot",
                        "chord_type",
                        "chord_tones",
                        "root",
                        "bass_note",
                        "alt_label",
                        "pedalend",
                    ]
                    feature_df.loc[ffill_mask, harmony_fill_columns] = (
                        feature_df.loc[ffill_mask, harmony_fill_columns]
                        .groupby(groupby_levels)
                        .ffill()
                    )
                if issubclass(cls, PhraseAnnotations):
                    group_intervals = get_index_intervals_for_phrases(
                        harmony_labels=feature_df,
                        group_cols=groupby_levels,
                        n_ante=feature_settings.get("n_ante"),
                        n_post=feature_settings.get("n_post"),
                        logger=self.logger,
                    )
                    ix_intervals = sum(group_intervals.values(), [])
                    feature_df = make_raw_phrase_df(
                        feature_df, ix_intervals, self.logger
                    )
                    feature_df = condense_pedal_points(feature_df)
                else:
                    feature_df = drop_rows_with_missing_values(
                        feature_df, feature_column_names, logger=self.logger
                    )
                feature_df = extend_harmony_feature(feature_df)
                feature_df = add_chord_tone_intervals(feature_df)
                feature_df = add_chord_tone_scale_degrees(feature_df)
        return feature_df


class MuseScoreMeasures(MuseScoreFacet, StructureFacet):
    _extractable_features = (FeatureName.Measures,)


class MuseScoreNotes(MuseScoreFacet, EventsFacet):
    _extractable_features = (FeatureName.Notes,)
