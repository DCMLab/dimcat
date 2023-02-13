"""Utility functions that are or might be used by several modules or useful in external contexts."""
import logging
from typing import Collection, List, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
import pandas as pd
from dimcat.base import Data, PipelineStep

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


T = TypeVar("T")
PotentiallyNested: TypeAlias = List[Union[T, "PotentiallyNested"]]


def grams(list_of_sequences: PotentiallyNested, n: int = 2) -> List[Tuple[T, ...]]:
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
        return list(zip(*(list_of_sequences[i:] for i in range(n))))


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


def transition_matrix(
    list_of_sequences: Optional[PotentiallyNested] = None,
    list_of_grams: Optional[List[Tuple[T, ...]]] = None,
    n=2,
    k=None,
    smooth: int = 0,
    normalize: bool = False,
    entropy: bool = False,
    excluded_symbols: Optional[List] = None,
    distinct_only: bool = False,
    sort: bool = False,
    percent: bool = False,
    decimals: Optional[int] = None,
) -> pd.DataFrame:
    """Returns a transition table from a list of symbols or a list of n-grams.
    You need to pass either ``list_of_sequences`` or ``list_of_grams``. If you pass the former, the latter will be
    created by calling the function :func:`grams`.

    Args:
        list_of_sequences:
            List of elements or nested list of elements between which the transitions are calculated. If the list
            is nested, bigrams are calculated recursively to exclude transitions between the lists. If you want
            to create the transition matrix from a list of n-grams directly, pass it as ``list_of_grams`` instead.
        list_of_grams: List of tuples being n-grams. If you want to have them computed from a list of sequences,
            pass it as ``list_of_sequences`` instead.
        n: If ``list_of_sequences`` is passed, the number of elements per n-gram tuple. Ignored otherwise.
        k: If specified, the transition matrix will show only the top k n-grams.
        smooth: If specified, this is the minimum value in the transition matrix.
        normalize: By default, absolute counts are shown. Pass True to normalize each row.
        entropy: Pass True to add a column showing the normalized entropy for each row.
        excluded_symbols: Any n-gram containing any of these symbols will be excluded.
        distinct_only: Pass True to exclude all n-grams consisting of identical elements only.
        sort:
            By default, the order of both index and columns follows the overall n-gram frequency.
            Pass True to sort them separately, i.e. each by their own frequencies.
        percent: Pass True to multiply the matrix by 100 before rounding to `decimals`
        decimals: To how many decimals you want to round the matrix, if at all.

    Returns:
        DataFrame with frequency statistics of (n-1) grams transitioning to all occurring last elements.
        The index is made up of strings corresponding to all but the last element of the n-grams,
        with the column index containing all last elements.
    """
    if list_of_grams is None:
        assert n > 1, f"Cannot print {n}-grams"
        list_of_grams = grams(list_of_sequences, n=n)
    elif list_of_sequences is not None:
        assert True, "Specify either list_of_grams or list_of_grams, not both."
    if len(list_of_grams) == 0:
        raise ValueError(
            "Unable to compute transition matrix from empty list of n-grams."
        )
    if excluded_symbols:
        list_of_grams = list(
            filter(lambda n: not any(g in excluded_symbols for g in n), list_of_grams)
        )
    if distinct_only:
        list_of_grams = list(
            filter(lambda tup: any(e != tup[0] for e in tup), list_of_grams)
        )
    ngrams = pd.Series(list_of_grams).value_counts()
    multiindex = pd.MultiIndex.from_tuples(ngrams.index)
    ngrams.index = multiindex
    consequent_level = multiindex.get_level_values(-1)
    if sort:
        contexts = ngrams.droplevel(-1)
        # add up the counts for identical contexts and consequents to sort them
        context_counts = (
            contexts.groupby(contexts.index).sum().sort_values(ascending=False)
        )
        context_levels = context_counts.index
        consequents = ngrams.copy()
        consequents.index = consequent_level
        consequent_counts = (
            consequents.groupby(consequents.index).sum().sort_values(ascending=False)
        )
        consequent_level = consequent_counts.index
    else:
        context_levels = multiindex.droplevel(-1).unique()
        consequent_level = consequent_level.unique()
    df = pd.DataFrame(smooth, index=context_levels, columns=consequent_level)

    for (*context, consequent), occurrences in ngrams.items():
        try:
            df.loc[tuple(context), [consequent]] = occurrences
        except Exception as e:
            logger.warning(
                f"Could not write the {occurrences} of the transition from {context}->{consequent}:\n{e}"
            )
            continue

    if normalize or entropy:
        df_norm = df.div(df.sum(axis=1), axis=0)

    if entropy:
        ic = np.log2(1 / df_norm)
        entropy = (ic * df_norm).sum(axis=1)
        # Identical calculations:
        # entropy = scipy.stats.entropy(df.transpose(),base=2)
        # entropy = -(df * np.log2(df)).sum(axis=1)
        if normalize:
            entropy /= np.log2(len(df.columns))
            df = pd.concat([entropy, df_norm], axis=1)
        else:
            df = pd.concat([entropy, df], axis=1)

    if k is not None:
        df = df.iloc[:k, :k]

    if percent:
        df.iloc[:, 0:] *= 100

    if decimals is not None:
        df.round(decimals, inplace=True)

    return df


def typestrings2types(typestrings: Union[str, Collection[str]]) -> Tuple[type]:
    """Turns one or several names of classes contained in this module into a
    tuple of references to these classes."""
    d_types = Data._registry
    ps_types = PipelineStep._registry
    if isinstance(typestrings, str):
        typestrings = [typestrings]
    result = []
    for typ in typestrings:
        if typ in d_types:
            result.append(d_types[typ])
        elif typ in ps_types:
            result.append(ps_types[typ])
        else:
            raise KeyError(
                f"Typestring '{typ}' does not correspond to a known subclass of PipelineStep or Data:\n"
                f"{ps_types}\n{d_types}"
            )
    return tuple(result)
