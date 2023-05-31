"""Utility functions that are or might be used by several modules or useful in external contexts."""
from __future__ import annotations

import logging
import os
from typing import Collection, Optional

import ms3
import pandas as pd

logger = logging.getLogger(__name__)


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


def grams(list_of_sequences, n=2):
    """Returns a list of n-gram tuples for given list. List can be nested.
    Use nesting to exclude transitions between pieces or other units.
    Uses: nest_level()

    """
    if nest_level(list_of_sequences) > 1:
        ngrams = []
        no_sublists = []
        for item in list_of_sequences:
            if isinstance(item, list):
                ngrams.extend(grams(item, n))
            else:
                no_sublists.append(item)
        if len(no_sublists) > 0:
            ngrams.extend(grams(no_sublists, n))
        return ngrams
    else:
        # if len(l) < n:
        #    print(f"{l} is too small for a {n}-gram.")
        # ngrams = [l[i:(i+n)] for i in range(len(l)-n+1)]
        ngrams = list(zip(*(list_of_sequences[i:] for i in range(n))))
        # convert to tuple of strings
        return [tuple(str(g) for g in gram) for gram in ngrams]


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


def interval_index2interval(ix):
    """Takes an interval index and returns the interval corresponding to [min(left), max(right))."""
    left = ix.left.min()
    right = ix.right.max()
    return pd.Interval(left, right, closed="left")


def replace_ext(filepath, new_ext):
    """Replace the extension of any given file path with a new one which can be given with or without a leading dot."""
    file, _ = os.path.splitext(filepath)
    if file.split(".")[-1] in ("resource", "datapackage", "package"):
        file = ".".join(file.split(".")[:-1])
    if new_ext[0] != ".":
        new_ext = "." + new_ext
    return file + new_ext


def get_object_value(obj, key, default):
    """Return obj[key] if possible, obj.key otherwise. Code copied from marshmallow.utils._get_value_for_key()"""
    if not hasattr(obj, "__getitem__"):
        return getattr(obj, key, default)

    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(obj, key, default)


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
    """
    path = ms3.resolve_dir(filepath)
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


def get_default_basepath():
    return os.getcwd()
