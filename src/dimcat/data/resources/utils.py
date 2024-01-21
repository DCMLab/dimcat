from __future__ import annotations

import json
import logging
import os
import re
import warnings
from collections import Counter
from functools import cache
from operator import itemgetter
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
    overload,
)
from zipfile import ZipFile

import frictionless as fl
import ms3
import numpy as np
import pandas as pd
from dimcat.base import (
    DimcatConfig,
    get_setting,
    is_instance_of,
    make_config_from_specs,
)
from dimcat.data.utils import make_fl_resource
from dimcat.dc_exceptions import (
    ResourceIsMissingPieceIndexError,
    ResourceNotProcessableError,
)
from dimcat.dc_warnings import ResourceWithRangeIndexUserWarning
from marshmallow.fields import Boolean
from ms3 import reduce_dataframe_duration_to_first_row
from ms3.expand_dcml import expand_labels
from numpy import typing as npt
from numpy._typing import NDArray

from .base import IX, D, FeatureName, S

module_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .base import SomeDataframe, SomeIndex
    from .dc import DimcatIndex, FeatureSpecs, Playthrough

TRUTHY_VALUES = Boolean.truthy
FALSY_VALUES = Boolean.falsy


DCML_CORPUS_NAMES: Dict[str, str] = {
    "ABC": "Beethoven String Quartets",
    "bach_en_fr_suites": "Bach Suites",
    "bach_solo": "Bach Solo",
    "bartok_bagatelles": "Bartok Bagatelles",
    "beethoven_piano_sonatas": "Beethoven Sonatas",
    "c_schumann_lieder": "C Schumann Lieder",
    "chopin_mazurkas": "Chopin Mazurkas",
    "corelli": "Corelli Trio Sonatas",
    "couperin_clavecin": "Couperin Clavecin",
    "couperin_concerts": "Couperin Concerts Royaux",
    "cpe_bach_keyboard": "CPE Bach Keyboard",
    "debussy_suite_bergamasque": "Debussy Suite Bergamasque",
    "dvorak_silhouettes": "Dvořák Silhouettes",
    "frescobaldi_fiori_musicali": "Frescobaldi Fiori Musicali",
    "gastoldi_baletti": "Gastoldi Baletti",
    "grieg_lyric_pieces": "Grieg Lyric Pieces",
    "handel_keyboard": "Handel Keyboard",
    "jc_bach_sonatas": "JC Bach Sonatas",
    "kleine_geistliche_konzerte": "Schütz Kleine Geistliche Konzerte",
    "kozeluh_sonatas": "Kozeluh Sonatas",
    "liszt_pelerinage": "Liszt Années",
    "mahler_kindertotenlieder": "Mahler Kindertotenlieder",
    "medtner_tales": "Medtner Tales",
    "mendelssohn_quartets": "Mendelssohn Quartets",
    "monteverdi_madrigals": "Monteverdi Madrigals",
    "mozart_piano_sonatas": "Mozart Piano Sonatas",
    "pergolesi_stabat_mater": "Pergolesi Stabat Mater",
    "peri_euridice": "Peri Euridice",
    "pleyel_quartets": "Pleyel Quartets",
    "poulenc_mouvements_perpetuels": "Poulenc Mouvements Perpetuels",
    "rachmaninoff_piano": "Rachmaninoff Piano",
    "ravel_piano": "Ravel Piano",
    "scarlatti_sonatas": "Scarlatti Sonatas",
    "schubert_dances": "Schubert Dances",
    "schubert_winterreise": "Schubert Winterreise",
    "schulhoff_suite_dansante_en_jazz": "Schulhoff Suite Dansante En Jazz",
    "schumann_kinderszenen": "R Schumann Kinderszenen",
    "schumann_liederkreis": "R Schumann Liederkreis",
    "sweelinck_keyboard": "Sweelinck Keyboard",
    "tchaikovsky_seasons": "Tchaikovsky Seasons",
    "wagner_overtures": "Wagner Overtures",
    "wf_bach_sonatas": "WF Bach Sonatas",
}


def align_with_grouping(
    df: pd.DataFrame, grouping: DimcatIndex | pd.MultiIndex, sort_index: bool = True
) -> pd.DataFrame:
    """Aligns a dataframe with a grouping index that has n levels such that the index levels of the  new dataframe
    start with the n levels of the grouping index and are followed by the remaining levels of the original dataframe.
    This is typically used to align a dataframe with feature information for many pieces with an index grouping
    piece names.
    """
    if not isinstance(grouping, pd.MultiIndex):
        if is_instance_of(grouping, "DimcatIndex"):
            grouping = grouping.index
        else:
            raise TypeError(
                f"Expected a MultiIndex or DimcatIndex, not {type(grouping)}"
            )
    df_levels = list(df.index.names)
    gr_levels = grouping.names
    if "piece" in gr_levels and "piece" not in df_levels:
        piece_level_position = infer_piece_col_position(df_levels)
        df = df.copy()
        df.index.rename("piece", level=piece_level_position, inplace=True)
        df_levels[piece_level_position] = "piece"
    if not set(df_levels).intersection(set(gr_levels)):
        raise ValueError(f"No shared levels between {df_levels!r} and {gr_levels!r}")
    df_aligned = join_df_on_index(df, grouping)
    if sort_index:
        return df_aligned.sort_index()
    return df_aligned


def apply_playthrough(
    feature_df: D,
    playthrough: Playthrough,
    logger: Optional[logging.Logger] = None,
) -> D:
    """Transform a dataframe based on the resource's :attr:`playthrough` setting."""
    if logger is None:
        logger = module_logger
    if playthrough == "RAW" or "volta" not in feature_df.columns:
        return feature_df
    if not playthrough == "SINGLE":
        raise NotImplementedError(f"Unknown Playthrough setting {playthrough!r}.")
    volta_values = feature_df.volta.unique()
    if 3 in volta_values:
        logger.info(
            "The dataframe has more than two alternative endings. The "
            "Playthrough.SINGLE setting drops all but the seconda volta."
        )
    keep_mask = feature_df.volta.isna() | feature_df.volta.eq(2)
    if keep_mask.all():
        logger.info("No alternative endings which would need to be dropped.")
        return feature_df
    drop_values = feature_df.loc[~keep_mask, "volta"].value_counts().to_dict()
    logger.debug(
        f"Values and occurrences of the dropped alternative endings:\n{drop_values}"
    )
    result = feature_df[keep_mask]
    if "quarterbeats_all_endings" in result.columns:
        return result.drop(columns="quarterbeats_all_endings")
    return result.copy()


def apply_slice_intervals_to_resource_df(
    df: pd.DataFrame,
    slice_intervals: pd.MultiIndex,
    qstamp_column_name: str = "quarterbeats",
    duration_column_name: str = "duration_qb",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = module_logger
    check_qstamp_columns(
        df=df,
        qstamp_column_name=qstamp_column_name,
        duration_column_name=duration_column_name,
        logger=logger,
    )
    *grouping_levels, slice_name = list(slice_intervals.names)
    interval_index_level = slice_intervals.get_level_values(-1)
    group2intervals: Dict[tuple, pd.IntervalIndex] = interval_index_level.groupby(
        slice_intervals.droplevel(-1)
    )

    sliced_dfs = {}
    for group, group_df in df.groupby(grouping_levels):
        ivls = group2intervals.get(group)
        if ivls is None:
            logger.info(
                f"{group!r}: No slice intervals present, group will not be omitted in the sliced resource."
            )
            continue
        time_spans, clean_group_df = get_time_spans_from_resource_df(
            df=group_df,
            qstamp_column_name=qstamp_column_name,
            duration_column_name=duration_column_name,
            dropna=True,
            return_df=True,
            logger=logger,
        )  # clean_group_df has no missing values in the columns used for computing time spans
        sliced_dfs[group] = overlapping_chunk_per_interval_cutoff_direct(
            df=clean_group_df.droplevel(grouping_levels),
            lefts=time_spans.start.values,
            rights=time_spans.end.values,
            intervals=ivls,
            qstamp_column_name=qstamp_column_name,
            duration_column_name=duration_column_name,
            logger=logger,
        )
    return pd.concat(sliced_dfs, names=grouping_levels)


def boolean_is_minor_column_to_mode(S: pd.Series) -> pd.Series:
    return S.map({True: "minor", False: "major"})


def check_qstamp_columns(
    df: D,
    qstamp_column_name: str,
    duration_column_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = module_logger
    if not all(c in df.columns for c in (qstamp_column_name, duration_column_name)):
        missing = [
            c for c in (qstamp_column_name, duration_column_name) if c not in df.columns
        ]
        plural = "s" if len(missing) > 1 else ""
        raise RuntimeError(
            f"Column{plural} not present in DataFrame: {', '.join(missing)}"
        )
    for col in (qstamp_column_name, duration_column_name):
        if df[col].isna().any():
            logger.warning(
                f"Column {col!r} has missing values which may lead to unintended results."
            )


def condense_dataframe_by_groups(
    df: pd.DataFrame,
    group_keys_series: pd.Series,
    logger: Optional[logging.Logger] = None,
):
    """Based on the given ``group_keys_series``, drop all rows but the first of each group and adapt the column
    'duration_qb' accordingly.


    Args:
        df:
            DataFrame to be reduced, expected to contain the column ``duration_qb``. In order to use the result as a
            segmentation, it should have a :obj:`pandas.IntervalIndex`.
        group_keys_series:
            Series with the same index as ``df`` that contains the group keys. If it contains NA values, the


    Returns:
        Reduced DataFrame with updated 'duration_qb' column and :obj:`pandas.IntervalIndex` on the first level
        (if present).

    """
    if logger is None:
        logger = module_logger
    if "duration_qb" not in df.columns:
        raise ValueError(f"DataFrame is missing the column 'duration_qb': {df.columns}")
    missing_duration_mask = df.duration_qb.isna()
    if missing_duration_mask.any():
        if missing_duration_mask.all():
            raise ValueError(
                "DataFrame contains only NA values in column 'duration_qb'."
            )
        logger.warning(
            f"DataFrame contains {missing_duration_mask.sum()} NA values in column 'duration_qb'. "
            f"Those rows will be dropped."
        )
        df = df[~missing_duration_mask]
        group_keys_series = group_keys_series[~missing_duration_mask]
    if group_keys_series.isna().any():
        logger.warning(
            f"The group_keys_series contains {group_keys_series.isna().sum()} NA values. The corresponding rows will "
            f"be dropped."
        )
    condensed = df.groupby(group_keys_series, group_keys=False).apply(
        reduce_dataframe_duration_to_first_row
    )
    return condensed


def condense_pedal_points(df):
    """Condenses pedal points into single rows. The duration of the pedal point is summed up and the chord is
    replaced by the pedal
    """
    group_start_mask = make_group_start_mask(df, df.index.names[:-1])
    pedal_point_mask = df.pedal.notna()
    shifted_pedals = df.pedal.shift().fillna("SENTINEL")
    shifted_pedals.loc[
        group_start_mask
    ] = "SENTINEL"  # make sure to separate a terminal pedal point (ending a piece)
    # from an initial one (beginning of next piece) on the same harmony (an extremely unlikely scenario)
    pedal_point_start_mask = (df.pedal != shifted_pedals).fillna(False)
    pedal_drop_mask = pedal_point_mask & ~pedal_point_start_mask
    expanded_pedal_harmonies = expand_labels(
        df.loc[pedal_point_start_mask, :"pedal"],
        column="pedal",
        propagate=False,
        skip_checks=True,
    )
    overwrite_with = expanded_pedal_harmonies.loc(axis=1)["chord":]
    overwrite_columns = overwrite_with.columns
    df.loc[pedal_point_start_mask, overwrite_columns] = overwrite_with
    df = df[~pedal_drop_mask]
    update_mask = df.pedal.notna() & ~make_groups_lasts_mask(
        df.index.to_frame(index=False), ["phrase_id", "phrase_component"]
    )  # this is not a clean solution; re-computation of duration for phrases needs an overhaul in conjunction with
    # solving the ToDo in facets.make_raw_phrase_df()
    update_duration_qb(df, update_mask)
    return df


def ensure_level_named_piece(
    index: pd.MultiIndex, recognized_piece_columns: Optional[Iterable[str]] = None
) -> Tuple[pd.MultiIndex, int]:
    """Ensures that the index has a level named "piece" by detecting alternative level names and renaming it in case it
     doesn't have one. Returns the index and the position of the piece level.


    Args:
        index: MultiIndex.
        recognized_piece_columns:
            Defaults to ("pieces", "fname", "fnames"). If other names are to be recognized as "piece" level, pass those.

    Returns:
        The same index or a copy with a renamed level.
        The position of the piece level.
    """
    level_names = index.names
    piece_level_position = infer_piece_col_position(
        level_names, recognized_piece_columns=recognized_piece_columns
    )
    if piece_level_position is None:
        resolved_arg = resolve_recognized_piece_columns_argument(
            recognized_piece_columns
        )
        raise ValueError(
            f"No level has any of the recognized 'piece' column names {resolved_arg!r}: {level_names!r}"
        )
    if level_names[piece_level_position] != "piece":
        return index.rename("piece", level=piece_level_position), piece_level_position
    return index, piece_level_position


def feature_specs2config(feature: FeatureSpecs) -> DimcatConfig:
    """Converts a feature specification to a DimcatConfig.

    Raises:
        TypeError:
            If the specs cannot be resolved to a :class:`DimcatConfig` that describes a Feature.
    """
    return make_config_from_specs(feature, "Feature")


def features_argument2config_list(
    features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
    allowed_configs: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
) -> List[DimcatConfig]:
    if features is None:
        return []
    if is_instance_of(features, (MutableMapping, "Feature", FeatureName, str)):
        features = [features]
    configs = []
    for specs in features:
        configs.append(make_config_from_specs(specs))
    if allowed_configs is not None:
        check_configs_against_allowed_configs(configs, allowed_configs)
    return configs


def check_configs_against_allowed_configs(
    configs: DimcatConfig | Iterable[DimcatConfig],
    allowed_configs: Optional[FeatureSpecs | Iterable[FeatureSpecs]],
    allow_subclasses: bool = True,
) -> None:
    """Matches configs against allowed configs and raises as soon as any pair does not match. Two
    configs match if they have the same dtype and any overlapping key has the same value.

    Args:
        configs: Config(s) to be checked.
        allowed_configs: The function raises if any of the ``configs`` does not match with any of these.
        allow_subclasses:
            If True (default), ``configs`` dtypes are allowed to be subclasses of the ``allowed_configs`` dtypes.

    Raises:
        ResourceNotProcessableError when any of the configs doesn't match with any of the allowed configs.
    """
    if isinstance(configs, DimcatConfig):
        configs = [configs]
    allowed_configs = features_argument2config_list(allowed_configs)
    covariant = allow_subclasses
    for configs in configs:
        if not any(
            configs.matches(allowed, covariant=covariant) for allowed in allowed_configs
        ):
            raise ResourceNotProcessableError(configs.options_dtype)


def drop_rows_with_missing_values(
    df: D,
    column_names: List[str],
    how: Literal["any", "all"] = "any",
    logger: Optional[logging.Logger] = None,
) -> D:
    """Drop rows with missing values in the specified columns. If nothing is to be dropped, the identical
    dataframe is returned, not a copy.
    """
    if logger is None:
        logger = module_logger
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
            f"The dataframe contains no fully defined objects based on the "
            f"columns {column_names}."
        )
    n_dropped = drop_mask.sum()
    if n_dropped:
        df = df[~drop_mask].copy()
        logger.info(
            f"Dropped {n_dropped} rows from the dataframe that pertain to segments following the last "
            f"cadence label in the piece."
        )
    return df


T = TypeVar("T")


def fl_fields2pandas_params(fields: List[fl.Field]) -> Tuple[dict, dict, list]:
    """Convert frictionless Fields to pd.read_csv() parameters 'dtype', 'converters' and 'parse_dates'."""
    dtype = {}
    converters = {}
    parse_dates = []
    for field in fields:
        if not field.type or field.type == "any":
            continue
        if field.type == "string":
            if pattern_constraint := field.constraints.get("pattern"):
                if pattern_constraint == ms3.FRACTION_REGEX:
                    converters[field.name] = ms3.safe_frac
                elif pattern_constraint == ms3.EDTF_LIKE_YEAR_REGEX:
                    # year number or '..'
                    converters[field.name] = ms3.safe_int
                elif pattern_constraint == ms3.INT_ARRAY_REGEX:
                    # a sequence of numbers
                    converters[field.name] = ms3.str2inttuple
                elif pattern_constraint == ms3.KEYSIG_DICT_REGEX:
                    converters[field.name] = ms3.str2keysig_dict
                elif pattern_constraint == ms3.TIMESIG_DICT_REGEX:
                    converters[field.name] = ms3.str2timesig_dict
                # ToDo: achieve this by creating a pd.IntervalIndex.from_arrays() or a MultiIndex.set_levels()
                # elif pattern_constraint == ms3.SLICE_INTERVAL_REGEX:
                #     converters[field.name] = str2pd_interval
                else:
                    raise NotImplementedError(
                        f"What is the dtype for a string with a pattern constraint of "
                        f"{pattern_constraint!r} (field {field.name})?"
                    )
            else:
                dtype[field.name] = "string"
        elif field.type == "integer":
            if field.required:
                dtype[field.name] = int
            else:
                dtype[field.name] = "Int64"
        elif field.type == "number":
            dtype[field.name] = float
        elif field.type == "boolean":
            converters[field.name] = value2bool
        elif field.type == "date":
            parse_dates.append(field.name)
        elif field.type == "array":
            converters[field.name] = str2inttuple
        # missing (see https://specs.frictionlessdata.io/table-schema)
        # - object (i.e. JSON/ a dict)
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
    return dtype, converters, parse_dates


@cache
def get_corpus_display_name(repo_name: str) -> str:
    """Looks up a repository name in the CORPUS_NAMES constant. If not present,
    the repo name is returned as title case.
    """
    name = DCML_CORPUS_NAMES.get(repo_name, "")
    if name == "":
        name = " ".join(s.title() for s in repo_name.split("_"))
    return name


def get_existing_normpath(fl_resource) -> str:
    """Get the normpath of a frictionless resource, raising an exception if it does not exist.

    Args:
        fl_resource:
            The frictionless resource. If its basepath is not specified, the filepath is tried
            relative to the current working directory.

    Returns:
        The absolute path of the frictionless resource.

    Raises:
        FileNotFoundError: If the normpath does not exist.
        ValueError: If the resource has no path or no column_schema.
    """
    if fl_resource.normpath is None:
        if not fl_resource.path:
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


@overload
def get_time_spans_from_resource_df(
    df: pd.DataFrame,
    qstamp_column_name: str,
    duration_column_name: str,
    round: Optional[int],
    to_float: bool,
    dropna: bool,
    return_df: Literal[False],
    logger: Optional[logging.Logger],
) -> pd.DataFrame:
    ...


@overload
def get_time_spans_from_resource_df(
    df: pd.DataFrame,
    qstamp_column_name: str,
    duration_column_name: str,
    round: Optional[int],
    to_float: bool,
    dropna: bool,
    return_df: Literal[True],
    logger: Optional[logging.Logger],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...


def get_time_spans_from_resource_df(
    df,
    qstamp_column_name: str = "quarterbeats",
    duration_column_name: str = "duration_qb",
    round: Optional[int] = None,
    to_float: bool = True,
    dropna: bool = False,
    return_df: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a dataframe with start ('left') and end ('right') positions of the events represented by this
    resource's rows.

    Args:
        df:
        qstamp_column_name: Column from which to retrieve start positions.
        duration_column_name: Column from which to retrieve durations to be added to the start positions.
        round:
            To how many decimal places to round the intervals' boundary values. Setting a value automatically sets
            ``to_float=True``.
        to_float:
            By default (True), the returned time span values are floats. Set False to leave values as they are
            after adding the columns, e.g. as fractions. If ``round`` is specified, however, this has no effect since
            the values are rounded to floats anyway.
        dropna:
            By default (False), rows with missing values are ignored and the result will include missing values for
            them. Pass True to drop rows with missing values. In this case you may also want to set ``return_df=True``.
        return_df:
            Pass True if you want to return the original dataframe as well, especially when ``dropna=True``.
        logger:

    Returns:
        A dataframe with columns ``start`` and ``end``.
        If ``return_df=True``, the input dataframe is returned as used for computing the time spans.
    """
    if logger is None:
        logger = module_logger
    check_qstamp_columns(df, qstamp_column_name, duration_column_name, logger=logger)
    available_mask = (
        df[qstamp_column_name].notna()
        & (df[qstamp_column_name] != "")
        & df[duration_column_name].notna()
        & (df[duration_column_name] != "")
    )
    if available_mask.sum() == 0:
        raise ValueError("No dimensions found for computing time spans.")
    some_are_missing = not available_mask.all()
    if some_are_missing:
        n_missing = (~available_mask).sum()
        msg = f"{n_missing} rows are coming without without time spans"
        if dropna:
            msg += " and have been dropped"
        msg += f":\n{df.index[~available_mask].to_list()}"
        logger.warning(msg)
        if not dropna:
            original_index = df.index
        df = df[available_mask].copy()
    start = df[qstamp_column_name]
    duration_col = df[duration_column_name]
    end = start + duration_col
    if to_float or round is not None:
        start = start.astype(float)
        end = end.astype(float)
        if round is not None:
            start, end = start.round(round), end.round(round)
    result = pd.DataFrame(
        {
            "start": start,
            "end": end,
            # duration_column_name: duration_col
        },
        index=df.index,
    )
    if some_are_missing and not dropna:
        result = result.reindex(original_index)
    if return_df:
        return result, df
    return result


def infer_piece_col_position(
    column_name: List[str],
    recognized_piece_columns: Optional[Iterable[str]] = None,
) -> Optional[int]:
    """Infer the position of the piece column in a list of column names."""
    recognized_piece_columns = resolve_recognized_piece_columns_argument(
        recognized_piece_columns
    )
    if recognized_piece_columns[0] != "piece":
        recognized_piece_columns = ["piece"] + [col for col in recognized_piece_columns]
    for name in recognized_piece_columns:
        try:
            return column_name.index(name)
        except ValueError:
            continue
    return


def infer_schema_from_df(
    df: SomeDataframe,
    include_index_levels: bool = True,
    allow_integer_names: bool = True,
    **kwargs,
) -> fl.Schema:
    """Infer a frictionless.Schema from a dataframe.

    This function partially copies ms3.utils.frictionless_helpers.get_schema().

    Args:
        df:
        include_index_levels:
            If False (default), the index levels are not described, assuming that they will not be written to disk
            (otherwise, validation error). Set to True to add all index levels to the described columns and,
            in addition, to make them the ``primaryKey`` (which, in frictionless, implies the constraints "required" &
            "unique").
        **kwargs:
            Arbitrary key-value pairs that will be added to the frictionless schema descriptor as "custom" metadata.

    Returns:

    """
    column_names = df.columns.to_list()
    if isinstance(column_names[0], tuple):
        if allow_integer_names:
            column_names = [
                ", ".join(str(name) for name in col) for col in column_names
            ]
        else:
            try:
                column_names = [", ".join(col) for col in column_names]
            except TypeError:
                raise TypeError(
                    f"Column names are tuples but not all elements are strings: {column_names}. Set "
                    f"allow_integer_names=True to convert all values to strings"
                )
    if include_index_levels:
        index_levels = list(df.index.names)
        column_names = index_levels + column_names
    else:
        index_levels = None
    if allow_integer_names:
        column_names = list(map(str, column_names))
    n_columns = len(column_names)
    n_unique = len(set(column_names))
    if n_unique < n_columns:
        non_unique = {
            name: occ for name, occ in Counter(column_names).items() if occ > 1
        }
        raise RuntimeError(
            f"The following columns are non-unique:\n{pformat(non_unique)}"
        )
    descriptor = make_frictionless_schema_descriptor(
        column_names=column_names,
        primary_key=index_levels,
        **kwargs,
    )
    return fl.Schema(descriptor)


def join_df_on_index(
    df: pd.DataFrame,
    index: DimcatIndex | pd.MultiIndex,
    how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
) -> pd.DataFrame:
    if is_instance_of(index, "DimcatIndex"):
        index = index.index
    # change left <-> right because this function uses the .join() method of the empty dataframe
    if how == "left":
        how = "right"
    if how == "right":
        how = "left"
    if df.columns.nlevels > 1:
        # workaround because pandas enforces column indices to have same number of levels for merging
        column_index = pd.MultiIndex.from_tuples(df.columns, names=df.columns.names)
        grouping_df = pd.DataFrame(index=index, columns=column_index)
        df_aligned = grouping_df.join(df, how=how, lsuffix="_")
    else:
        grouping_df = pd.DataFrame(index=index)
        df_aligned = grouping_df.join(
            df,
            how=how,
        )
    # makes sure that the joined-on index levels are the leftmost ones
    index_level_order = list(grouping_df.index.names)
    index_level_order += [
        level for level in df.index.names if level not in index_level_order
    ]
    if df_aligned.index.names != index_level_order:
        df_aligned = df_aligned.reorder_levels(index_level_order)
    return df_aligned


def load_fl_resource(
    fl_resource: fl.Resource,
    normpath: Optional[str] = None,
    index_col: Optional[int | str | Iterable[int | str]] = None,
    usecols: Optional[int | str | Iterable[int | str]] = None,
) -> SomeDataframe:
    """Load a dataframe from a :obj:`frictionless.Resource`.

    Args:
        fl_resource: The resource whose normpath points to a file on the local file system.
        normpath:
            If not specified, the normpath of the resource is used, which is not always reliable because its
            own basepath property is half-heartedly maintained.
        index_col: Column(s) to be used as index levels, overriding the primary key specified in the resource's schema.
        usecols: If only a subset of the specified fields is to be loaded, the names or positions of the subset.

    Returns:
        The loaded dataframe loaded with the dtypes resulting from converting the schema fields via
        :func:`fl_fields2pandas_params`.
    """
    if not normpath:
        normpath = get_existing_normpath(fl_resource)
    field_names = fl_resource.schema.field_names
    index_col_names = resolve_columns_argument(index_col, field_names)
    usecols_names = resolve_columns_argument(usecols, field_names)
    if usecols_names is None:
        usecols_names = field_names
    if index_col_names is None:
        if fl_resource.schema.primary_key:
            index_col_names = fl_resource.schema.primary_key
        else:
            warnings.warn(
                f"Resource {fl_resource.name!r} has no primary key and no index_col was given. "
                f"Dataframe will come with a RangeIndex.",
                ResourceWithRangeIndexUserWarning,
            )
    if index_col_names is not None:
        missing_col_names = [
            col_name for col_name in index_col_names if col_name not in usecols_names
        ]
        usecols_names = missing_col_names + usecols_names
    usecols_fields = [fl_resource.schema.get_field(name) for name in usecols_names]
    dtypes, converters, parse_dates = fl_fields2pandas_params(usecols_fields)
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
            file,
            sep="\t",
            usecols=lambda x: x in usecols_names,
            parse_dates=parse_dates,
            dtype=dtypes,
            converters=converters,
        )
    except Exception:
        module_logger.warning(
            f"Error executing\n"
            f"pd.read_csv({file!r},\n"
            f"            sep='\\t',\n"
            f"            usecols={usecols_names!r},\n"
            f"            parse_dates={parse_dates!r},\n"
            f"            dtype={dtypes!r},\n"
            f"            converters={converters!r})"
        )
        raise
    if index_col_names:
        return dataframe.set_index(index_col_names)
    else:
        dataframe.index.name = "i"
        return dataframe


def load_index_from_fl_resource(
    fl_resource: fl.Resource,
    index_col: Optional[int | str | List[int | str]] = None,
    recognized_piece_columns: Iterable[str] = ("piece", "pieces", "fname", "fnames"),
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

    Returns:
        The specified or inferred index column(s) as a (Multi)Index object.

    Raises:
        FileNotFoundError: If the normpath of the resource does not exist.
        ValueError: If the resource doesn't yield a normpath or the index columns cannot be inferred from it
            based on the schema.
    """
    _ = get_existing_normpath(fl_resource)  # raises if normpath doesn't exist
    schema = fl_resource.schema
    right_most_column = None
    if recognized_piece_columns:
        recognized_piece_columns = tuple(recognized_piece_columns)
        if len(recognized_piece_columns) > 0:
            right_most_column = recognized_piece_columns[0]
    if index_col is not None:
        if isinstance(index_col, (int, str)):
            index_col = [index_col]
    elif schema.primary_key:
        index_col = schema.primary_key
        module_logger.debug(
            f"Loading index columns {index_col!r} from {fl_resource.name!r} based on the schema's "
            f"primary key."
        )
    elif right_most_column:
        module_logger.debug(
            f"Resource {fl_resource.name!r} has no primary key, trying to detect {right_most_column!r} column."
        )
        piece_col_position = infer_piece_col_position(
            schema.field_names,
            recognized_piece_columns=recognized_piece_columns,
        )
        if piece_col_position is None:
            raise ResourceIsMissingPieceIndexError(fl_resource.name, right_most_column)
        index_col = list(range(piece_col_position + 1))
        module_logger.debug(
            f"Loading index columns {index_col!r} from {fl_resource.name!r} based on the recognized "
            f"piece column {schema.field_names[piece_col_position]!r}."
        )
    else:
        raise ValueError(
            f"Resource {fl_resource.name!r} has no primary key and neither index_col nor recognized_piece_columns "
            f"were specified."
        )
    dataframe = load_fl_resource(fl_resource, index_col=index_col, usecols=index_col)
    return dataframe.index


def make_adjacency_groups(
    S: pd.Series,
    groupby=None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.Series, Dict[int, Any]]:
    """Turns a Series into a Series of ascending integers starting from 1 that reflect groups of successive
    equal values.

    This is a simplified variant of ms3.adjacency_groups()

    Args:
      S: Series in which to group identical adjacent values with each other.
      groupby:
        If not None, the resulting grouper will start new adjacency groups according to this groupby.
        This is a way, for example, to ensure no group overlaps piece boundaries even if there are
        adjacent identical values.

    Returns:
      A series with increasing integers that can be used for grouping.
      A dictionary mapping the integers to the grouped values.

    """
    if logger is None:
        logger = module_logger
    if groupby is None:
        beginnings = make_adjacency_mask(S)
    else:
        beginnings = S.groupby(groupby, group_keys=False).apply(make_adjacency_mask)
    groups = beginnings.cumsum()
    names = dict(enumerate(S[beginnings], 1))
    try:
        return pd.to_numeric(groups).astype("Int64"), names
    except TypeError:
        logger.warning(f"Erroneous outcome while computing adjacency groups: {groups}")
        return groups, names


def make_adjacency_mask(
    S: pd.Series,
) -> pd.Series:
    """Turns a Series into a Boolean Series that is True for the first value of each group of successive equal
    values.
    """
    assert not S.isna().any(), "Series must not contain NA values."
    beginnings = (S != S.shift()).fillna(True)
    return beginnings


def make_boolean_mask_from_set_of_tuples(
    index: DimcatIndex | pd.MultiIndex,
    tuples: Set[tuple],
    levels: Optional[Iterable[int]] = None,
) -> pd.Index[bool]:
    """Returns a boolean mask for the given tuples based on index tuples formed from integer positions of the index
    levels to subselect.

    Args:
        index: Index (of the dataframe) you want to subselect from using the returned boolean mask.
        tuples:
        levels:

            * If None, the first n levels of the index are used, where n is the length of the selection tuples.
            * If an iterable of level name strings or level position integers, they are used to create for each row a
              tuple to compare against the selected tuples.

    Returns:
        A boolean mask of the same length as the index, where True indicates that the corresponding index tuple is
        contained in the selection tuples.

    Raises:
        TypeError: If tuples is not a set.
        ValueError: If tuples is empty.
        ValueError: If the index has less levels than the selection tuples.
        ValueError: If levels is not None and has a different length than the selection tuples.
    """
    if not isinstance(tuples, set):
        raise TypeError(f"tuples must be a set, not {type(tuples)}")
    if len(tuples) == 0:
        raise ValueError("tuples must not be empty")
    random_tuple = next(iter(tuples))
    n_selection_levels = len(random_tuple)
    if index.nlevels < n_selection_levels:
        raise ValueError(
            f"index has {index.nlevels} levels, but {n_selection_levels} levels were specified for selection."
        )
    if levels is None:
        # select the first n levels
        next_to_each_other = True
        levels = tuple(range(n_selection_levels))
    else:
        # clean up levels argument
        is_int, is_str = isinstance(levels, int), isinstance(levels, str)
        if (is_int or is_str) and n_selection_levels > 1:
            # only the first level was specified, select its n-1 right neighbours, too
            if is_str:
                position = index.names.index(levels)
            else:
                position = levels
            levels = tuple(position + i for i in range(n_selection_levels))
            next_to_each_other = True
        else:
            levels = resolve_levels_argument(levels, index.names)
            if len(levels) != n_selection_levels:
                raise ValueError(
                    f"The selection tuples have length {n_selection_levels}, but {len(levels)} levels were specified: "
                    f"{levels}."
                )
            next_to_each_other = all(b == a + 1 for a, b in zip(levels, levels[1:]))
    if n_selection_levels == index.nlevels:
        drop_levels = None
    else:
        drop_levels = tuple(i for i in range(index.nlevels) if i not in levels)
    if drop_levels:
        index = index.droplevel(drop_levels)
    if next_to_each_other:
        return index.isin(tuples)
    tuple_maker = itemgetter(*levels)
    return index.map(lambda index_tuple: tuple_maker(index_tuple) in tuples)


def make_frictionless_schema_descriptor(
    column_names: Iterable[str],
    primary_key: Optional[Iterable[str]] = None,
    **custom_data,
) -> dict:
    """Creates a frictionless schema descriptor from a list of column names and a primary key.

    This function is a duplicate of ms3.utils.frictionless_helpers.make_frictionless_schema_descriptor() and the
    translation of column names into frictionless fields (with type and description) falls back to
    ms3.utils.frictionless_helpers.column_name2frictionless_field().

    Args:
        column_names:
        primary_key:
        **custom_data:

    Returns:

    """
    fields = []
    if primary_key:
        for ix_level, column in zip(primary_key, column_names):
            if ix_level != column:
                raise ValueError(
                    f"primary_key {primary_key} does not match column_names {column_names[:len(primary_key)]}"
                )
    for column_name in column_names:
        field = ms3.column_name2frictionless_field(column_name)
        if "type" not in field:
            raise ValueError(
                f"column_name2frictionless_field({column_name!r}) = {field} (missing 'type'!)"
            )
        fields.append(field)
    descriptor = dict(fields=fields)
    if primary_key:
        descriptor["primaryKey"] = list(primary_key)
    if len(custom_data) > 0:
        descriptor.update(custom_data)
    return descriptor


def make_group_start_mask(df: D, groupby) -> npt.NDArray[bool]:
    """Returns a boolean mask where the beginning of each group is marked with True. This is useful only when the
    groups already came in groups within the dataframe in the first place.
    """
    group_start_idx = np.array([idx[0] for idx in df.groupby(groupby).indices.values()])
    group_start_mask = np.zeros(len(df), bool)
    group_start_mask[group_start_idx] = True
    return group_start_mask


def make_groups_lasts_mask(feature_df: D | S, groupby=None) -> npt.NDArray[bool]:
    """Returns a boolean mask where each row that comes last in one of the groups is marked as True. This is
    useful only when the groups already came in groups within the dataframe in the first place.
    Instead of a dataframe with groupby columns you may also pass a Series with None.
    """
    if isinstance(feature_df, pd.Series):
        groups_last_idx = np.array(
            [idx[-1] for idx in feature_df.groupby(feature_df).indices.values()]
        )
    else:
        groups_last_idx = np.array(
            [idx[-1] for idx in feature_df.groupby(groupby).indices.values()]
        )
    result = np.zeros(feature_df.shape[0], bool)
    result[groups_last_idx] = True
    return result


def make_index_from_grouping_dict(
    grouping: Dict[str, Iterable[tuple]],
    level_names=("group_name", "corpus", "piece"),
    sort=False,
    raise_if_multiple_membership: bool = False,
) -> pd.MultiIndex:
    """Creates a MultiIndex from a dictionary with grouped tuples.

    Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is sorted. Set False to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.

    Returns:
        A MultiIndex with the given names and the tuples from the grouping dictionary.
    """
    if not isinstance(grouping, dict):
        raise TypeError(f"grouping must be a dict, not {type(grouping)}")
    if raise_if_multiple_membership:
        all_members = []
        for members in grouping.values():
            all_members.extend(members)
        if len(all_members) != len(set(all_members)):
            c = Counter(all_members)
            multiple_members = [member for member, count in c.items() if count > 1]
            raise ValueError(f"Multiple membership detected: {multiple_members}")
    index_tuples = [
        (group,) + piece for group, pieces in grouping.items() for piece in pieces
    ]
    if sort:
        index_tuples = sorted(index_tuples)
    return pd.MultiIndex.from_tuples(index_tuples, names=level_names)


def make_phrase_start_mask(df) -> npt.NDArray[bool]:
    """Based on the "phrase_id" index level, make a mask that is True for the first row of each mask."""
    phrase_ids = df.index.get_level_values("phrase_id").to_numpy()
    phrase_start_mask = phrase_ids != np.roll(phrase_ids, 1)
    return phrase_start_mask


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
    resource = make_fl_resource(name, **options)
    resource.type = "table"
    return resource


def merge_columns_into_one(
    df: D,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> S:
    """Merge all columns of a dataframe into a single column.

    Args:
        df: Dataframe to reduce.
        join_str:
            By default (None), the resulting columns contain tuples. If you want them to contain strings,
            you may pass

            - True to concatenate the tuple values for a given n-gram component separated by ", " --
              yielding strings that look like tuples without parentheses
            - False to concatenate without any string in-between the values
            - a string to be used as the separator between the tuple values.

        fillna:
            Pass a value to replace all missing values with it.

    Returns:
        A series containing tuples or strings.
    """
    if fillna is not None:
        df = df.fillna(fillna)
    result = pd.Series(df.itertuples(index=False, name=None), index=df.index)
    if join_str is not None:
        join_str = resolve_join_str_argument(join_str)
        result = ms3.transform(result, tuple2str, join_str=join_str)
    return result


def merge_ties(
    df: D,
    return_dropped: bool = False,
    perform_checks: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """In a note list, merge tied notes to single events with accumulated durations.
    Input dataframe needs columns ['duration', 'tied', 'midi', 'staff']. This
    function does not handle correctly overlapping ties on the same pitch since
    it doesn't take into account the notational layers ('voice').

    Copied from ms3, to be developed further.



    Args:
        df:
        return_dropped:
        perform_checks:
        logger:

    Returns:

    """
    if logger is None:
        logger = module_logger

    def merge(df):
        vc = df.tied.value_counts()
        try:
            if vc[1] != 1 or vc[-1] != 1:
                logger.warning(f"More than one 1 or -1:\n{vc}")
        except KeyError:
            logger.warning(f"Inconsistent 'tied' values while merging: {vc}")
        ix = df.iloc[0].name
        dur = df.duration.sum()
        drop = df.iloc[1:].index.to_list()
        return pd.Series({"ix": ix, "duration": dur, "dropped": drop})

    def merge_notes(staff_midi):
        staff_midi["chunks"] = (staff_midi.tied == 1).astype(int).cumsum()
        t = staff_midi.groupby("chunks", group_keys=False).apply(merge)
        return t.set_index("ix")

    if not df.tied.notna().any():
        return df
    df = df.copy()
    notna = df.loc[df.tied.notna(), ["duration", "tied", "midi", "staff"]]
    if perform_checks:
        before = notna.tied.value_counts()
    new_dur = (
        notna.groupby(["staff", "midi"], group_keys=False)
        .apply(merge_notes)
        .sort_index()
    )
    try:
        df.loc[new_dur.index, "duration"] = new_dur.duration
    except Exception:
        print(new_dur)
        raise
    if return_dropped:
        df.loc[new_dur.index, "dropped"] = new_dur.dropped
    df = df.drop(new_dur.dropped.sum())
    if perform_checks:
        after = df.tied.value_counts()
        assert (
            before[1] == after[1]
        ), f"Error while merging ties. Before:\n{before}\nAfter:\n{after}"
    return df


def nan_eq(a, b):
    """Returns True if a and b are equal or both null. Works on two Series or two elements."""
    return (a == b) | (pd.isnull(a) & pd.isnull(b))


def overlapping_chunk_per_interval_cutoff_direct(
    df: pd.DataFrame,
    lefts: NDArray,
    rights: NDArray,
    intervals: pd.IntervalIndex,
    qstamp_column_name: str = "quarterbeats",
    duration_column_name: str = "duration_qb",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """The heart of a slicing operation, which returns a dataframe that corresponds to the input dataframe sliced
    by the intervals present in the ``intervals`` :obj:`pandas.IntervalIndex`, which will be included as the first
    index level of the result dataframe.

    Args:
        df: DataFrame to be sliced.
        lefts: Same-length array expressing the start point of every row.
        rights: Same-length array expressing the end point (exclusive) of every row.
        qstamp_column_name:
            Name of the column in which qstamp (offset from the timeline's origin) is to be found.
        duration_column_name:
            Name of the column in the chunk dfs where the new event durations will be stored as floats. Defaults to
            "duration_qb", resulting in the existing values being updated.
        intervals:
            The pairs are interpreted as left-closed, right-open intervals that demarcate the boundaries of the
            returned DataFrame chunks. These intervals are assumed to be non-overlapping and monotonically
            increasing, which allows us to speed up this expensive operation.

    Returns:
        Concatenation of the dataframe chunks corresponding to each of the given interval. The first index level of the
        resulting dataframe is a :obj`pandas.IntervalIndex` which corresponds to the ``intervals``.
    """
    if logger is None:
        logger = module_logger
    check_qstamp_columns(
        df=df,
        qstamp_column_name=qstamp_column_name,
        duration_column_name=duration_column_name,
        logger=logger,
    )
    if not intervals.is_non_overlapping_monotonic:
        logger.warning(
            f"Intervals are not non-overlapping and/or not monotonically increasing:\n{intervals}"
        )
    n = len(df.index)
    chunks = {}  # slice_iv -> mask
    current_start_mask = np.ones(n, dtype=bool)
    for interval in intervals:
        # never again check events ending before the current interval's start
        l, r = interval.left, interval.right
        not_ending_before_l = rights >= l
        lefts = lefts[not_ending_before_l]
        rights = rights[not_ending_before_l]
        current_start_mask[current_start_mask] = not_ending_before_l
        starting_before_r = r > lefts
        not_ending_on_l_except_empty = (rights != l) | (lefts == l)
        overlapping = starting_before_r & not_ending_on_l_except_empty
        bool_mask = current_start_mask.copy()
        bool_mask[current_start_mask] = overlapping
        new_lefts, new_rights = lefts[overlapping], rights[overlapping]
        starting_before_l, ending_after_r = (new_lefts < l), (new_rights > r)
        chunk = df[bool_mask].copy()
        if starting_before_l.sum() > 0 or ending_after_r.sum() > 0:
            new_lefts[starting_before_l] = l
            new_rights[ending_after_r] = r
            chunk[qstamp_column_name] = new_lefts
            chunk[duration_column_name] = (new_rights - new_lefts).round(5)
        chunks[interval] = chunk
    level_names = [intervals.name] + df.index.names
    return pd.concat(chunks, names=level_names)


def resolve_columns_argument(
    columns: Optional[int | str | Iterable[int | str]],
    column_names: List[str],
) -> Optional[List[str]]:
    """Resolve the columns argument of a load function to a list of column names.

    Args:
        columns:
            A list of integer position and/or column names. Can be mixed but integers will always
            be interpreted as positions.
        column_names: List of column names to choose from.

    Returns:
        The resolved list of column names. None if ``columns`` is None.

    Raises:
        ValueError: If ``columns`` contains duplicate column names.
    """
    if columns is None:
        return
    result = []
    if isinstance(columns, (int, str)):
        columns = [columns]
    for str_or_int in columns:
        if isinstance(str_or_int, int):
            result.append(column_names[str_or_int])
        elif str_or_int in column_names:
            result.append(str_or_int)
        else:
            pass
    result_set = set(result)
    if len(result_set) != len(result):
        raise ValueError(f"Duplicate column names in {columns}.")
    return result


def resolve_levels_argument(
    levels: Optional[int | str | Iterable[int | str]],
    level_names: List[str],
    inverse: bool = False,
) -> Optional[Tuple[int]]:
    """Turns a selection of index levels into a list of positive level positions."""
    if levels is None:
        return
    result = []
    nlevels = len(level_names)
    if isinstance(levels, (int, str)):
        levels = [levels]
    for str_or_int in levels:
        if isinstance(str_or_int, int):
            if str_or_int < 0:
                as_int = nlevels + str_or_int
            else:
                as_int = str_or_int
        else:
            as_int = level_names.index(str_or_int)
        if as_int < 0 or as_int >= nlevels:
            raise ValueError(
                f"Level {str_or_int!r} not found in level names {level_names}."
            )
        result.append(as_int)
    result_set = set(result)
    if len(result_set) != len(result):
        raise ValueError(f"Duplicate level names in {levels}.")
    if inverse:
        result = [i for i in range(nlevels) if i not in result]
    return tuple(result)


def resolve_recognized_piece_columns_argument(
    recognized_piece_columns: Optional[Iterable[str]] = None,
) -> List[str]:
    """Resolve the recognized_piece_columns argument by replacing None with the default value."""
    if recognized_piece_columns is None:
        return get_setting("recognized_piece_columns")
    else:
        return list(recognized_piece_columns)


def resolve_join_str_argument(
    join_str: Optional[bool | str | Tuple[bool | str, ...]]
) -> Optional[str]:
    """Helper function that resolves a join_str argument to a string or None by replacing boolean values with the
    defaults ", " for True and "" for False.
    """
    if join_str is None:
        return
    if not isinstance(join_str, str):
        if join_str is True:
            join_str = ", "
        elif join_str is False:
            join_str = ""
        else:
            raise TypeError(
                f"join_str must be a string or a boolean, got {join_str!r} ({type(join_str)})"
            )
    return join_str


def safe_row_tuple(row: Iterable[str]) -> str | Literal[pd.NA]:
    """Join the given strings together separated by ', ' but catch TypeErrors by returning pd.NA instead."""
    try:
        return ", ".join(row)
    except TypeError:
        return pd.NA


def store_json(
    data: dict, filepath: str, indent: int = 2, make_dirs: bool = True, **kwargs
):
    """Serialize object to file.

    Args:
        data: Nested structure of dicts and lists.
        filepath: Path to the text file to (over)write.
        indent: Prettify the JSON layout. Default indentation: 2 spaces
        make_dirs: If True (default), create the directory if it does not exist.
        **kwargs: Keyword arguments passed to :meth:`json.dumps`.
    """
    kwargs = dict(indent=indent, **kwargs)
    if make_dirs:
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)


def str2inttuple(s):
    """Non-strict version of :func:`ms3.str2inttuple` which does not fail on non-integer values."""
    return ms3.str2inttuple(s, strict=False)


def str2pd_interval(s: str) -> pd.Interval:
    """Function produces only left-closed, right-open intervals."""
    left, right = re.match(ms3.SLICE_INTERVAL_REGEX, s).groups()
    return pd.Interval(left=float(left), right=float(right), closed="left")


def subselect_multiindex_from_df(
    df: D,
    tuples: DimcatIndex | Iterable[tuple],
    levels: Optional[int | str | List[int | str]] = None,
) -> pd.DataFrame:
    """Returns a copy of a subselection of the dataframe based on the union of its index tuples (or subtuples)
    and the given tuples.

    Args:
        df: Dataframe of which to return a subset of rows.
        tuples: Tuples to match against df's MultiIndex. Can be a MultiIndex because set(tuples) works on that, too.
        levels:

            * If None, the first n levels of the index are used, where n is the length of the selection tuples.
            * If an iterable of level name strings or level position integers, they are used to create for each row a
              tuple to compare against the selected tuples.

    Returns:

    """
    tuple_set = set(tuples)
    if not len(tuple_set):
        raise ValueError("Received 0 tuples")
    random_tuple = next(iter(tuple_set))
    if not isinstance(random_tuple, tuple):
        raise TypeError(
            f"Pass an iterable of tuples. A randomly selected element had type {type(random_tuple)!r}."
        )
    mask = make_boolean_mask_from_set_of_tuples(df.index, tuple_set, levels)
    return df[mask].copy()


def transpose_notes_to_c(notes: D) -> D:
    """Transpose the columns 'tpc' and 'midi' in a way that they reflect the local key as if it was C major/minor. This
    operation is typically required for creating pitch class profiles.
    Uses: :py:func:`ms3.transform`, :py:func:`ms3.name2fifths`, :py:func:`ms3.roman_numeral2fifths`

    Args:
        notes: DataFrame that has at least the columns ['globalkey', 'localkey', 'tpc', 'midi'].

    Returns:
         A new dataframe with the columns 'local_tonic_name', 'fifths_over_local_tonic', and 'midi_in_c'
         where the latter two correspond to the original columns 'tpc' and 'midi' but transposed in such a way that
         ``fifths_over_local_tonic == 0`` and ``midi_in_c % 12 == 0`` for all pitches that match the local tonic.
         E.g. for the local key A major/minor, each pitch A will have tpc=0 and midi % 12 = 0).
    """
    transpose_by = ms3.transform(notes.globalkey, ms3.name2fifths) + ms3.transform(
        notes, ms3.roman_numeral2fifths, ["localkey", "globalkey_is_minor"]
    )
    fifths_over_local_tonic = notes.tpc - transpose_by
    midi_transposition = ms3.transform(transpose_by, ms3.fifths2pc)
    # For transpositions up to a diminished fifth, move pitches up,
    # for larger intervals, move pitches down.
    midi_transposition.where(
        midi_transposition <= 6, midi_transposition % -12, inplace=True
    )
    midi_in_c = notes.midi - midi_transposition
    local_tonic = pd.concat(
        [transpose_by.rename("local_tonic"), notes.localkey_is_minor], axis=1
    )
    local_tonic = ms3.transform(
        local_tonic,
        ms3.fifths2name,
        dict(fifths="local_tonic", minor="localkey_is_minor"),
    )
    return pd.concat(
        [
            local_tonic.rename("local_tonic_name"),
            fifths_over_local_tonic.rename("fifths_over_local_tonic"),
            midi_in_c.rename("midi_in_c"),
        ],
        axis=1,
    )


def tuple2str(
    tup: tuple,
    join_str: Optional[str] = ", ",
    recursive: bool = True,
    keep_parentheses: bool = False,
) -> str:
    """Used for turning n-gram components into strings, e.g. for display on plot axes.

    Args:
        tup: Tuple to be returned as string.
        join_str:
            String to be interspersed between tuple elements. If None, result is ``str(tup)`` and ``recursive`` is
            ignored.
        recursive:
            If True (default) tuple elements that are tuples themselves will be joined together recursively, using the
            same ``join_str`` (except when it's None). Inner tuples always keep their parentheses.
        keep_parentheses: If False (default), the outer parentheses are removed. Pass True to keep them in the string.

    Returns:
        A string representing the tuple.
    """
    try:
        if join_str is None:
            result = str(tup)
            if keep_parentheses:
                return result
            return result[1:-1]
        if recursive:
            result = join_str.join(
                tuple2str(e, join_str=join_str, keep_parentheses=True)
                if isinstance(e, tuple)
                else str(e)
                for e in tup
            )
        else:
            result = join_str.join(str(e) for e in tup)
    except TypeError:
        return str(tup)
    if keep_parentheses:
        return f"({result})"
    return result


def update_duration_qb(
    df: D,
    update_mask: Optional[npt.NDArray[bool]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Replaces the 'duration_qb' column in the given DataFrame with a new one that updates the values by subtracting
    subsequent 'quarterbeats' values. If ``update_mask`` is specified, only values for which the mask is True are
    updated. Otherwise, all values are updated.
    """
    if logger is None:
        logger = module_logger
    updated_durations = (df.quarterbeats.shift(-1) - df.quarterbeats).astype(float)
    if update_mask is None:
        updated_duration_qb_column = updated_durations
    else:
        updated_duration_qb_column = updated_durations.where(
            update_mask, df.duration_qb
        )
    updated_mask = updated_duration_qb_column != df.duration_qb
    logger.debug(
        f"{updated_mask.sum()} values have been updated in the 'duration_qb' for phrase annotations."
    )
    df.loc[:, "duration_qb"] = updated_duration_qb_column


def value2bool(value: str | float | int | bool) -> bool | str | float | int:
    """Identical with ms3.value2bool"""
    if value in TRUTHY_VALUES:
        return True
    if value in FALSY_VALUES:
        return False
    if isinstance(value, str):
        try:
            converted = float(value)
        except Exception:
            return value
        if converted in TRUTHY_VALUES:
            return True
        if converted in FALSY_VALUES:
            return False
    return value


# region PhraseData helpers


def append_index_levels(
    old_index: IX,
    *new_level: IX | S | D,
    drop_levels: Optional[Literal[False], str | int | Iterable[str | int]] = None,
) -> IX:
    """
    Replace index levels by optionally dropping an arbitrary number and concatenating the new level(s) to the right.
    """
    if drop_levels:
        old_index = old_index.droplevel(drop_levels)
    new_levels = [
        level.reset_index(drop=True)
        if isinstance(level, (pd.Series, pd.DataFrame))
        else level.to_frame(index=False)
        for level in new_level
    ]
    new_index_df = pd.concat([old_index.to_frame(index=False)] + new_levels, axis=1)
    new_index = pd.MultiIndex.from_frame(new_index_df)
    return new_index


def insert_index_level(old_index: IX, new_level: IX | S | D, position: int) -> IX:
    """
    Replace index levels by optionally dropping an arbitrary number and concatenating the new level(s) to the right.
    """
    if isinstance(new_level, (pd.Series, pd.DataFrame)):
        new_level = new_level.reset_index(drop=True)
    else:
        # should be a DimcatIndex or pd.Index (or pd.MultiIndex)
        new_level = new_level.to_frame(index=False)
    new_index_df = old_index.to_frame(index=False)
    new_index_df = pd.concat(
        [
            new_index_df.iloc(axis=1)[:position],
            new_level,
            new_index_df.iloc(axis=1)[position:],
        ],
        axis=1,
    )
    new_index = pd.MultiIndex.from_frame(new_index_df)
    return new_index


def make_groupwise_range_index_from_groups(idx: pd.Index) -> npt.NDArray[int]:
    """Turns adjacency groups into integer ranges starting from 0."""
    arr = idx.to_numpy()
    start_mask = arr != np.roll(
        arr, 1
    )  # position 0 correct only when last != first (because of how roll works)
    return make_range_index_from_boolean_mask(start_mask)


def make_range_index_from_boolean_mask(
    inner_start_mask: npt.NDArray[bool],
    outer_start_mask: Optional[npt.NDArray[bool]] = None,
) -> npt.NDArray[int]:
    """Creates an index with the same length as the given boolean mask, that restarts counting from every True entry.
    The behaviour changes depending on whether outer_start_mask is given or not. That's how the function is used
    by :meth:`PhraseData._regroup_phrases` to create both the inner and the outer index level. The function is
    indifferent to the value of the first entry in the mask(s).

    The algorithm builds on Warren Weckesser's approach via https://stackoverflow.com/a/20033438


    Args:
        inner_start_mask:
        outer_start_mask:

    Returns:

    """
    if outer_start_mask is None:
        increments = np.asarray(
            ~inner_start_mask, int
        )  # increment for every False (non-start)
        (reset_index,) = np.where(inner_start_mask)
    else:
        increments = np.asarray(
            inner_start_mask, int
        )  # increment only for every True (start)
        (reset_index,) = np.where(outer_start_mask)

    # ensure the same behaviour regardless of the value of the first element
    increments[0] = 0  # always start counting at 0
    if len(reset_index) == 0 or reset_index[0] != 0:
        np.insert(reset_index, 0, 0)

    if outer_start_mask is None:
        reset_starts_by = (
            reset_index[1:] - reset_index[:-1]
        )  # last range value + 1 = distance between starts
    else:
        outer_mask_increment_counts = increments.cumsum()[outer_start_mask]
        reset_starts_by = (
            outer_mask_increment_counts[1:] - outer_mask_increment_counts[:-1]
        )  # last range value + 1 = number of inner increments between outer starts

    increments[reset_index[1:]] = 1 - reset_starts_by
    increments.cumsum(out=increments)
    return increments


def make_regrouped_stage_index(
    df: D,
    grouping: S,
    level_names: Tuple[str, str] = ("stage", "substage"),
) -> D:
    """Returns a dataframe that corresponds to the two new (stage) index levels that :func:`regroup_phrase_stages`
    incorporates.
    """
    assert len(grouping.shape) == 1, "Expecting a Series."
    phrase_start_mask = make_phrase_start_mask(df)
    substage_start_mask = (
        (grouping != grouping.shift()).fillna(True).to_numpy(dtype=bool)
    ) | phrase_start_mask
    substage_level = make_range_index_from_boolean_mask(substage_start_mask)
    # make new stage level that restarts at phrase starts and increments at substage starts
    stage_level = make_range_index_from_boolean_mask(
        substage_start_mask, phrase_start_mask
    )
    # create index levels as dataframe in order to concatenate them to existing levels
    primary, secondary = level_names
    new_index = pd.DataFrame({primary: stage_level, secondary: substage_level})
    return new_index


def make_multiindex_for_unstack(idx: pd.Index, level_name: str = "i") -> pd.MultiIndex:
    """Turns an index that contains adjacency groups (adjacent entries having the same value) into
    a 2-level MultiIndex where the new level represents an individual integer range for each group,
    starting at 0.
    """
    old_level_name = idx.name
    groupwise_ranges = make_groupwise_range_index_from_groups(idx)
    result = pd.MultiIndex.from_arrays(
        [idx, groupwise_ranges], names=[old_level_name, level_name]
    )
    return result


def regroup_phrase_stages(
    df: D,
    grouping: S,
    level_names: Tuple[str, str] = ("stage", "substage"),
):
    """Insert a grouping column and replace the last index level with a new primary and secondary index accordingly.
    The primary level increments at the beginning of each group, the secondary level increments at every row,
    restarting at the beginning of each group. For example, a grouping ["a", "a", "a", "b", "c", "c"] results
    in the index [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)].


    Args:
        grouping:
            A Series with the same index as the (raw) phrase_df, containing the grouping criterion. Adjacent equal
            values are grouped together.
        level_names: Names of the two index levels.

    Returns:
        A reindexed copy of the phrase data.
    """
    new_index = make_regrouped_stage_index(df, grouping, level_names)
    result_df = pd.concat([grouping, df], axis=1)
    result_df.index = append_index_levels(result_df.index, new_index, drop_levels=-1)
    return result_df


def drop_duplicated_ultima_rows(phrase_annotations_df: D) -> D:
    """Used by the :class:`PhraseDataAnalyzer` to drop the last row of each phrase's body component when
    ``drop_duplicated_ultima_rows`` is True.
    """
    groups_last_mask = make_groups_lasts_mask(
        phrase_annotations_df, ["phrase_id", "phrase_component"]
    )
    body_mask = (
        phrase_annotations_df.index.get_level_values("phrase_component").to_numpy()
        == "body"
    )
    body_last_row_mask = body_mask & groups_last_mask
    return phrase_annotations_df[~body_last_row_mask].copy()


phraseComponents: TypeAlias = Literal["ante", "body", "codetta", "post"]


def transform_phrase_data(
    phrase_df,
    columns: str | List[str] = "chord",
    components: phraseComponents | List[phraseComponents] = "body",
    drop_levels: bool | int | str | Iterable[int | str] = False,
    reverse: bool = False,
    level_name: str = "i",
):
    """Returns a dataframe containing the requested phrase components and harmony columns.

    Args:
        phrase_df: PhraseAnnotations dataframe.
        columns:
            Column(s) to include in the result.
        components:
            Which of the four phrase components to include, ∈ {'ante', 'body', 'codetta', 'post'}.
        drop_levels:
            Can be a boolean or any level specifier accepted by :meth:`pandas.MultiIndex.droplevel()`.
            If False (default), all levels are retained. If True, only the phrase_id level and
            the ``level_name`` are retained. In all other cases, the indicated (string or
            integer) value(s) must be valid and cause one of the index levels to be dropped.
            ``level_name`` cannot be dropped. Dropping 'phrase_id' will likely lead to an
            exception if a :class:`PhraseData` object will be displayed in WIDE format.
        reverse:
            Pass True to reverse the order of harmonies so that each phrase's last label comes
            first.
        level_name:
            Defaults to 'i', which is the name of the original level that will be replaced
            by this new one. The new one represents the individual integer range for each
            phrase, starting at 0.

    Returns:
        Dataframe representing partial information on the selected phrases.
    """
    result = phrase_df.loc[pd.IndexSlice[:, :, :, components], columns].copy()
    if reverse:
        result = result[::-1]
    phrase_ids = result.index.get_level_values("phrase_id")
    if drop_levels is True:
        new_index = make_multiindex_for_unstack(phrase_ids, level_name=level_name)
    else:
        new_level = pd.Series(
            make_groupwise_range_index_from_groups(phrase_ids), name=level_name
        )
        old_index = result.index.droplevel(-1)
        new_index = append_index_levels(old_index, new_level, drop_levels=drop_levels)
    result.index = new_index
    return result


# endregion PhraseData helpers
