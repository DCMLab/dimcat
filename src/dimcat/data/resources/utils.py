from __future__ import annotations

import logging
import os
from collections import Counter
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import frictionless as fl
import pandas as pd
from dimcat.base import get_setting

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base import DimcatIndex, SomeDataframe, SomeIndex


def align_with_grouping(
    df: pd.DataFrame, grouping: pd.MultiIndex, sort_index: bool = True
) -> pd.DataFrame:
    """Aligns a dataframe with a grouping index that has n levels such that the index levels of the  new dataframe
    start with the n levels of the grouping index and are followed by the remaining levels of the original dataframe.
    This is typically used to align a dataframe with feature information for many pieces with an index grouping
    piece names.
    """
    df_levels = list(df.index.names)
    gr_levels = grouping.names
    if "piece" in gr_levels and "piece" not in df_levels:
        piece_level_position = infer_piece_col_position(df_levels)
        df = df.copy()
        df.index.rename("piece", level=piece_level_position, inplace=True)
        df_levels[piece_level_position] = "piece"
    shared_levels = set(df_levels).intersection(set(gr_levels))
    if len(shared_levels) == 0:
        raise ValueError(f"No shared levels between {df_levels!r} and {gr_levels!r}")
    grouping_df = pd.DataFrame(index=grouping)
    df_aligned = grouping_df.join(
        df,
        how="left",
    )
    level_order = gr_levels + [level for level in df_levels if level not in gr_levels]
    result = df_aligned.reorder_levels(level_order)
    if sort_index:
        return result.sort_index()
    return result


def check_rel_path(rel_path, basepath):
    if rel_path.startswith(".."):
        raise ValueError(
            f"{rel_path!r} points outside the basepath {basepath!r} which is not allowed."
        )
    if rel_path.startswith(f".{os.sep}") and len(rel_path) > 2:
        rel_path = rel_path[2:]
    return rel_path


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


def fl_fields2pandas_params(fields: List[fl.Field]) -> Tuple[dict, dict, list]:
    """Convert frictionless Fields to pd.read_csv() parameters 'dtype', 'converters' and 'parse_dates'."""
    dtype = {}
    converters = {}
    parse_dates = []
    for field in fields:
        if not field.type or field.type == "any":
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
        elif field.type == "date":
            parse_dates.append(field.name)
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
    return dtype, converters, parse_dates


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


def infer_schema_from_df(df: SomeDataframe) -> fl.Schema:
    """Infer a frictionless.Schema from a dataframe."""
    return fl.Schema.describe(df)


def load_fl_resource(
    fl_resource: fl.Resource,
    index_col: Optional[int | str | List[int | str]] = None,
    usecols: Optional[Union[int, str, List[int | str]]] = None,
) -> SomeDataframe:
    """Load a dataframe from a :obj:`frictionless.Resource`.

    Args:
        fl_resource: The resource whose normpath points to a file on the local file system.
        index_col: Column(s) to be used as index levels, overriding the primary key specified in the resource's schema.
        usecols: If only a subset of the specified fields is to be loaded, the names or positions of the subset.

    Returns:
        The loaded dataframe loaded with the dtypes resulting from converting the schema fields via
        :func:`fl_fields2pandas_params`.
    """
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
            logger.warning(
                f"Resource {fl_resource.name!r} has no primary key and no index_col was given. "
                f"Dataframe will come with a RangeIndex."
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
            usecols=usecols_names,
            parse_dates=parse_dates,
            dtype=dtypes,
            converters=converters,
        )
    except Exception:
        logger.warning(
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
        logger.debug(
            f"Loading index columns {index_col!r} from {fl_resource.name!r} based on the schema's "
            f"primary key."
        )
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
        logger.debug(
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

            * If None, the last n levels of the index are used, where n is the length of the selection tuples.
            * If an iterable of integers, they are interpreted as level positions and used to create for each row a
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
            # only the first level was specified, select its n-1 right neightbours, too
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


def make_index_from_grouping_dict(
    grouping: Dict[str, List[tuple]],
    level_names=("group_name", "corpus", "piece"),
    sort=False,
    raise_if_multiple_membership: bool = False,
) -> pd.MultiIndex:
    """Creates a MultiIndex from a dictionary with grouped tuples.

    Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names: Names for the levels of the MultiIndex, i.e. one for the group level and one per grouped level.
        sort: By default the returned MultiIndex is sorted. Set False to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.

    Returns:
        A MultiIndex with the given names and the tuples from the grouping dictionary.
    """
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


def make_rel_path(path: str, start: str):
    """Like os.path.relpath() but ensures that path is contained within start."""
    rel_path = os.path.relpath(path, start)
    return check_rel_path(rel_path, start)


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


def resolve_columns_argument(
    columns: Optional[int | str | List[int | str]],
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
        else:
            if str_or_int not in column_names:
                raise ValueError(
                    f"Column {str_or_int!r} not found in field names {column_names}."
                )
            result.append(str_or_int)
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
