"""Utility functions that are or might be used by several modules or useful in external contexts."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import (
    Any,
    Collection,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    overload,
)
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dimcat.base import FriendlyEnum
from dimcat.data.base import AbsolutePathStr
from tqdm.auto import tqdm

# region constants

logger = logging.getLogger(__name__)

FRICTIONLESS_NAME_PATTERN = (
    r"^([-a-z0-9._/])+$"  # from frictionless.settings import NAME_PATTERN
)
FRICTIONLESS_INVERSE = r"[^-a-z0-9._/]"

# endregion constants
# region classes


class SortOrder(FriendlyEnum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    NONE = "NONE"


# endregion classes
# region functions


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

    Raises:
        FileNotFoundError: If the file does not exist and must_exist is True.
        ValueError: If the file does not have one of the specified extensions, if any.
    """
    path = resolve_path(filepath)
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


def check_name(name: str) -> None:
    """Check if a name is valid according to frictionless.

    Raises:
        ValueError: If the name is not valid.
    """
    if not re.match(FRICTIONLESS_NAME_PATTERN, name):
        raise ValueError(
            f"Name can only contain [a-z], [0-9], [-._/], and no spaces: {name!r}"
        )


def clean_index_levels(pandas_obj):
    """Remove index levels "IDs", "corpus" and "fname", if redundant."""
    idx = pandas_obj.index
    drop = []
    if idx.nlevels > 1 and "IDs" in idx.names:
        drop.append("IDs")
    if idx.names.count("corpus") > 1:
        drop.append("corpus")
    if idx.names.count("fname") > 1:
        drop.append("fname")
    if len(drop) > 0:
        # for each name, store the integer of the last level with that name
        name2level = {name: level for level, name in enumerate(idx.names)}
        drop_levels = [name2level[name] for name in drop]
        return pandas_obj.droplevel(drop_levels)
    return pandas_obj


def get_composition_year(metadata_dict):
    """The logic for getting a composition year out of the given metadata dictionary."""
    start = (
        metadata_dict["composed_start"] if "composed_start" in metadata_dict else None
    )
    end = metadata_dict["composed_end"] if "composed_end" in metadata_dict else None
    if pd.isnull(start) and pd.isnull(end):
        raise LookupError("Metadata do not include composition dates.")
    if pd.isnull(start):
        return end
    if pd.isnull(end):
        return start
    return round((end + start) / 2, ndigits=1)


def get_middle_composition_year(
    metadata: pd.DataFrame,
    composed_start_column: str = "composed_start",
    composed_end_column: str = "composed_end",
) -> pd.Series:
    """Returns the middle of the composition year range."""
    composed_start = pd.to_numeric(metadata[composed_start_column], errors="coerce")
    composed_end = pd.to_numeric(metadata[composed_end_column], errors="coerce")
    composed_start.fillna(composed_end, inplace=True)
    composed_end.fillna(composed_start, inplace=True)
    return (composed_start + composed_end) / 2


def get_object_value(obj, key, default):
    """Return obj[key] if possible, obj.key otherwise. Code copied from marshmallow.utils._get_value_for_key()"""
    if not hasattr(obj, "__getitem__"):
        return getattr(obj, key, default)

    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(obj, key, default)


def grams(lists_of_symbols, n=2, to_string: bool = False):
    """Returns a list of n-gram tuples for given list. List can be nested.

    Use nesting to exclude transitions between pieces or other units.

    """
    if nest_level(lists_of_symbols) > 1:
        ngrams = []
        no_sublists = []
        for item in lists_of_symbols:
            if isinstance(item, list):
                ngrams.extend(grams(item, n, to_string=to_string))
            else:
                no_sublists.append(item)
        if len(no_sublists) > 0:
            ngrams.extend(grams(no_sublists, n, to_string=to_string))
        return ngrams
    else:
        # if len(l) < n:
        #    print(f"{l} is too small for a {n}-gram.")
        # ngrams = [l[i:(i+n)] for i in range(len(l)-n+1)]
        ngrams = list(zip(*(lists_of_symbols[i:] for i in range(n))))
        # convert to tuple of strings
        if to_string:
            return [tuple(str(g) for g in gram) for gram in ngrams]
        else:
            return ngrams


def interval_index2interval(ix):
    """Takes an interval index and returns the interval corresponding to [min(left), max(right))."""
    left = ix.left.min()
    right = ix.right.max()
    return pd.Interval(left, right, closed="left")


def is_uri(path: str) -> bool:
    """Solution from https://stackoverflow.com/a/38020041"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def make_extension_regex(
    extensions: Iterable[str],
    enforce_initial_dot: bool = False,
) -> re.Pattern:
    """Turns file extensions into a regular expression."""
    if not extensions:
        return re.compile(".*")
    if isinstance(extensions, str):
        extensions = [extensions]
    else:
        extensions = list(extensions)
    if enforce_initial_dot:
        dot = r"\."
        regex = f"(?:{'|'.join(dot + e.lstrip('.') for e in extensions)})$"
    else:
        regex = f"(?:{'|'.join(extensions)})$"
    return re.compile(regex, re.IGNORECASE)


def make_suffix(*params):
    """Turns the passed parameter values into a suffix string.

    Parameters
    ----------
    params : str or Collection or number
        Parameters to turn into string components of the returned suffix. None values are ignored.
        Pairs of the form (str, <param>) are treated specially in that the initial str
        is treated as a prefix of the string component unless <param> is an empty/None/False value.

    Returns
    -------
    str
        A suffix string where the passed values are joined together, separated by '-'.

    Examples
    --------
    >>> make_suffix('str', None, False, {0, 1.})
    '-str-False-0|1.0'
    >>> make_suffix(['collection', 0], ('zero', 0), ('prefix', 1), ('flag', True))
    '-collection|0-prefix1-flag'
    """
    param_strings = []
    for p in params:
        if p is None:
            continue
        as_str = ""
        if isinstance(p, tuple):
            if len(p) == 2 and isinstance(p[0], str):
                as_str, p = p
                if not p:  # this catches 0, None, '', False etc.
                    continue
                if isinstance(p, bool):  # param is True
                    param_strings.append(as_str)
                    continue
        if isinstance(p, str):
            as_str += p
        elif isinstance(p, Collection):
            if len(p) == 0:
                continue
            as_str += "|".join(str(e) for e in p)
        else:
            as_str += str(p)
        param_strings.append(as_str)
    return "-".join(param_strings)


def make_transition_matrix(
    nested_sequences: Optional[list] = None,
    ngrams: Optional[List[tuple]] = None,
    n: int = 2,
    k: Optional[int] = None,
    smooth: int = 0,
    normalize: bool = False,
    IC: bool = False,
    excluded_grams: Optional[Any] = None,
    distinct_only: bool = False,
    sort: bool = False,
    percent: bool = False,
    decimals: Optional[int] = None,
):
    """Returns a transition table from a list of symbols or from a list of n-grams.

    Column index is the last item of grams, row index the n-1 preceding items.

    Args:
        nested_sequences:
            List of elements between which the transitions are calculated. If specified, ``ngrams`` must be None.
            List can be nested.
        ngrams: List of tuples being n-grams. If specified, ``nested_sequences`` must be None.
        n: Make n-grams. Only relevant if ``nested_sequences`` is specified.
        k: Number of rows and columns that you want to keep. Defaults to all.
        smooth: Initial count value of all transitions
        normalize: Set to True to divide every row by the sum of the row.
        IC: Set True to calculate information content.
        excluded_grams:
            Elements you want to exclude from the table. All ngrams containing at least one of the elements will be
            filtered out.
        distinct_only: if True, n-grams consisting only of identical elements are filtered out
        sort: By default, the indices are ordered by gram frequency. Pass True to sort by bigram counts.
        percent: Pass True to multiply the matrix by 100 before rounding to ``decimals``
        decimals: To how many decimals you want to round the matrix.

    Returns:
        For each (n-1) previous elements (index), the number or proportion of transitions to each possible following
        element (columns).
    """
    if ngrams is None:
        assert n > 0, f"Cannot print {n}-grams"
        ngrams = grams(nested_sequences, n=n, to_string=True)
    elif nested_sequences is not None:
        assert True, "Specify either l or gs, not both."

    if excluded_grams:
        ngrams = list(filter(lambda n: not any(g in excluded_grams for g in n), ngrams))
    if distinct_only:
        ngrams = list(filter(lambda tup: any(e != tup[0] for e in tup), ngrams))
    ngrams = pd.Series(ngrams).value_counts()
    if n > 2:
        ngrams.index = [(" ".join(t[:-1]), t[-1]) for t in ngrams.index.tolist()]
    context = pd.Index(set([ix[0] for ix in ngrams.index]))
    consequent = pd.Index(set([ix[1] for ix in ngrams.index]))
    df = pd.DataFrame(smooth, index=context, columns=consequent)

    for (cont, cons), n_gram_count in ngrams.items():
        try:
            df.loc[cont, cons] += n_gram_count
        except Exception:
            continue

    if k is not None:
        sort = True

    if sort:
        h_sort = list(df.max().sort_values(ascending=False).index.values)
        v_sort = list(df.max(axis=1).sort_values(ascending=False).index.values)
        df = df[h_sort].loc[v_sort]
    else:
        frequency = df.sum(axis=1).sort_values(ascending=False).index
        aux_index = frequency.intersection(df.columns, sort=False)
        aux_index = aux_index.union(
            df.columns.difference(frequency, sort=False), sort=False
        )
        df = df[aux_index].loc[frequency]

    SU = df.sum(axis=1)
    if normalize or IC:
        df = df.div(SU, axis=0)

    if IC:
        ic = np.log2(1 / df)
        ic["entropy"] = (ic * df).sum(axis=1)
        # ############# Identical calculations:
        # ic['entropy2'] = scipy.stats.entropy(df.transpose(),base=2)
        # ic['entropy3'] = -(df * np.log2(df)).sum(axis=1)
        df = ic
        if normalize:
            df["entropy"] = df["entropy"] / np.log2(len(df.columns) - 1)
    # else:
    #     df['total'] = SU

    if k is not None:
        df = df.iloc[:k, :k]

    if percent:
        df.iloc[:, :-1] *= 100

    if decimals is not None:
        df = df.round(decimals)

    return df


def make_valid_frictionless_name(name: str, replace_char="_") -> str:
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, not {type(name)}")
    name = name.lower()
    if not re.match(FRICTIONLESS_NAME_PATTERN, name):
        name = re.sub(FRICTIONLESS_INVERSE, replace_char, name)
    return name


def make_valid_frictionless_name_from_filepath(
    path: str, include_extension=True, replace_char="_"
) -> str:
    if include_extension:
        return make_valid_frictionless_name(path, replace_char=replace_char)
    name, _ = os.path.splitext(path)
    return make_valid_frictionless_name(name, replace_char=replace_char)


def nest_level(obj, include_tuples=False):
    """Recursively calculate the depth of a nested list."""
    if obj.__class__ != list:
        if include_tuples:
            if obj.__class__ != tuple:
                return 0
        else:
            return 0
    max_level = 0
    for item in obj:
        max_level = max(max_level, nest_level(item, include_tuples=include_tuples))
    return max_level + 1


def replace_ext(filepath, new_ext):
    """Replace the extension of any given file path with a new one which can be given with or without a leading dot."""
    file, _ = os.path.splitext(filepath)
    if file.split(".")[-1] in ("resource", "datapackage", "package"):
        file = ".".join(file.split(".")[:-1])
    if new_ext[0] != ".":
        new_ext = "." + new_ext
    return file + new_ext


@overload
def resolve_path(path: str) -> AbsolutePathStr:
    ...


@overload
def resolve_path(path: Literal[None]) -> None:
    ...


def resolve_path(path: str | Literal[None]) -> Optional[AbsolutePathStr]:
    """Resolves '~' to HOME directory and turns ``path`` into an absolute path."""
    if path is None:
        return None
    if isinstance(path, str):
        pass
    elif isinstance(path, Path):
        path = str(path)
    else:
        raise TypeError(f"Expected str or Path, got {type(path)}")
    if "~" in path:
        path = os.path.expanduser(path)
    else:
        path = os.path.abspath(path)
    path = path.rstrip("/\\")
    return AbsolutePathStr(path)


@overload
def treat_basepath_argument(path: str, other_logger) -> AbsolutePathStr:
    ...


@overload
def treat_basepath_argument(path: Literal[None], other_logger) -> None:
    ...


def treat_basepath_argument(
    basepath: str | Literal[None], other_logger: Optional[logging.Logger] = None
) -> Optional[AbsolutePathStr]:
    """Turns ``basepath`` into an absolute path and checks that it exists.

    Raises:
        NotADirectoryError: If ``basepath`` is not an existing directory.
    """
    if basepath is None:
        return
    basepath_arg = resolve_path(basepath)
    if not os.path.isdir(basepath_arg):
        raise NotADirectoryError(
            f"basepath {basepath_arg!r} is not an existing directory."
        )
    if other_logger is None:
        other_logger = logger
    other_logger.debug(f"The basepath been set to {basepath_arg!r}")
    return basepath_arg


@overload
def scan_directory(
    directory,
    extensions,
    file_re,
    folder_re,
    exclude_re,
    recursive,
    return_tuples: Literal[False],
    progress,
    exclude_files_only,
) -> Iterator[str]:
    ...


@overload
def scan_directory(
    directory,
    extensions,
    file_re,
    folder_re,
    exclude_re,
    recursive,
    return_tuples: Literal[True],
    progress,
    exclude_files_only,
) -> Iterator[Tuple[str, str]]:
    ...


def scan_directory(
    directory: str,
    extensions: Optional[str | Iterable[str]] = None,
    file_re: Optional[str] = None,
    folder_re: Optional[str] = None,
    exclude_re: str = r"^(\.|_)",
    recursive: bool = True,
    return_tuples: bool = False,
    progress: bool = False,
    exclude_files_only: bool = False,
) -> Iterator[str] | Iterator[Tuple[str, str]]:
    """Depth-first generator of filtered file paths in ``directory``.

    Args:
      directory: Directory to be scanned for files.
      extensions: File extensions to be included (with or without leading dot). Defaults to all extensions.
      file_re, folder_re:
          Regular expressions for filtering certain file names or folder names.
          The regEx are checked with search(), not match(), allowing for fuzzy search.
      exclude_re:
          Exclude files and folders (unless ``exclude_files_only=True``) containing this regular expression.
          Excludes files starting with a dot or underscore by default, prevent by setting to None or ''.
      recursive: By default, subdirectories are recursively scanned. Pass False to scan only ``dir``.
      return_tuples: By default, full file paths are returned. Pass True to return (path, name) tuples instead.
      progress: Pass True to display the progress (useful for large directories).
      exclude_files_only:
          By default, ``exclude_re`` excludes files and folder. Pass True to exclude only files matching the regEx.

    Yields:
      Full file path or, if ``return_tuples=True``, (path, file_name) pairs in random order.
    """
    if file_re is None:
        file_re = r".*"
    if folder_re is None:
        folder_re = r".*"
    extensions_regex = ".*" if extensions is None else make_extension_regex(extensions)

    def traverse(d):
        nonlocal counter

        def check_regex(reg, s, excl=exclude_re):
            try:
                passing = re.search(reg, s) is not None and re.search(excl, s) is None
            except Exception:
                logger.error(reg)
                raise
            return passing

        for dir_entry in os.scandir(d):
            name = dir_entry.name
            path = os.path.join(d, name)
            if dir_entry.is_dir():
                if not recursive:
                    continue
                if exclude_files_only:
                    if not check_regex(folder_re, name, excl="^$"):
                        continue
                else:
                    if not check_regex(folder_re, name):
                        continue
                for res in traverse(path):
                    yield res
            else:
                if pbar is not None:
                    pbar.update()

                if (
                    dir_entry.is_file()
                    and check_regex(extensions_regex, name)
                    and check_regex(file_re, name)
                ):
                    counter += 1
                    if pbar is not None:
                        pbar.set_postfix({"selected": counter})
                    if return_tuples:
                        yield d, name
                    else:
                        yield path

    if exclude_re is None or exclude_re == "":
        exclude_re = "^$"
    directory = resolve_path(directory)
    counter = 0
    if not os.path.isdir(directory):
        raise NotADirectoryError("Not an existing directory: " + directory)
    pbar = tqdm(desc="Scanning files", unit=" files") if progress else None
    return traverse(directory)


# endregion functions
