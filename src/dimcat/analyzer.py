"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from abc import ABC, abstractmethod

import pandas as pd
from ms3 import fifths2name

from .data import Data
from .pipeline import PipelineStep
from .utils import grams


class Analyzer(PipelineStep, ABC):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed."""


class FacetAnalyzer(Analyzer):
    """Analyzers that work on one particular type of DataFrames."""

    def __init__(self, once_per_group=False):
        """

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one result per group item.
            Set to True to instead compute only one for each group.
        """
        self.required_facets = []
        self.once_per_group = once_per_group
        self.config = {}
        """:obj:`dict`
        This dictionary stores the parameters to be passed to the compute() method."""
        self.group2pandas = None
        """:obj:`str`
        The name of the function that allows displaying one group's results as a single
        pandas object. See data.Corpus.convert_group2pandas()"""
        self.level_names = {"indices": "IDs"} if once_per_group else {}
        """:obj:`dict`
        Define {"indices": "index_level_name"} if the analysis is applied once per group,
        because the index of the DataFrame holding the processed data won't be showing the
        individual indices anymore.
        """

    @abstractmethod
    def compute(self, df):
        """Where the actual computation takes place."""

    def process_data(self, data: Data) -> Data:
        """Returns a copy of the Data object containing processed data."""
        processed = {}
        for group, dfs in data.iter_facet(
            self.required_facets[0], concatenate=self.once_per_group
        ):
            processed_group = {}
            for ID, df in dfs.items():
                key = "group_ids" if self.once_per_group else ID
                eligible, message = self.check(df)
                if not eligible:
                    print(f"{ID}: {message}")
                    continue
                processed_group[key] = self.compute(df, **self.config)
            if len(processed_group) == 0:
                print(f"Group '{group}' will be missing from the processed data.")
                continue
            processed[group] = processed_group
        result = data.copy()
        result.track_pipeline(self, group2pandas=self.group2pandas, **self.level_names)
        result.processed = processed
        return result


class NotesAnalyzer(FacetAnalyzer):
    def __init__(self, once_per_group=False):
        """Analyzers that work on notes tables.

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one result per group item.
            Set to True to instead compute only one for each group.
        """
        super().__init__(once_per_group=once_per_group)
        self.required_facets = ["notes"]


class ChordSymbolAnalyzer(FacetAnalyzer):
    def __init__(self, once_per_group=False):
        """Analyzers that work on expanded annotation tables.

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one result per group item.
            Set to True to instead compute only one for each group.
        """
        super().__init__(once_per_group=once_per_group)
        self.required_facets = ["expanded"]


class TPCrange(NotesAnalyzer):
    """Computes the range from the minimum to the maximum Tonal Pitch Class (TPC)."""

    def __init__(self, once_per_group=False):
        super().__init__(once_per_group=once_per_group)
        self.level_names["processed"] = "tpc_range"
        self.group2pandas = "group_of_values2series"

    @staticmethod
    def compute(df):
        """Computes the range from the minimum to the maximum Tonal Pitch Class (TPC).

        Parameters
        ----------
        df : pandas.DataFrame
            notes table with the column ``tpc``

        Returns
        -------
        int
        """
        return df.tpc.max() - df.tpc.min()


class PitchClassVectors(NotesAnalyzer):
    """Analyzer that groups notes by their pitch class and aggregates their durations."""

    def __init__(
        self,
        once_per_group=False,
        pitch_class_format="tpc",
        normalize=False,
        ensure_pitch_classes=None,
    ):
        """Analyzer that groups notes by their pitch class and aggregates their durations.

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one pitch class vector per group item.
            Set to True to instead compute only one for each group.
        pitch_class_format : :obj:`str`, optional
            | Defines the type of pitch classes.
            | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
            | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
            | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
            | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
        normalize : :obj:`bool`, optional
            By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
            the PCV for each slice.
        ensure_pitch_classes : :obj:`Iterable`, optional
            By default, pitch classes that don't appear don't appear. Pass a collection of pitch
            classes if you want to ensure their presence even if empty. For example, if
            ``pitch_class_format='pc'`` you could pass ``ensure_columns=range(12)``.
        """
        super().__init__(once_per_group=once_per_group)
        self.config = dict(
            pitch_class_format=pitch_class_format,
            normalize=normalize,
            ensure_pitch_classes=ensure_pitch_classes,
        )
        self.level_names["processed"] = pitch_class_format
        self.group2pandas = "group2dataframe_unstacked"

    @staticmethod
    def compute(
        notes,
        pitch_class_format="tpc",
        normalize=False,
        ensure_pitch_classes=None,
    ):
        """Group notes by their pitch class and aggregate their durations.

        Parameters
        ----------
        notes : :obj:`pandas.DataFrame`
            Note table to be transformed into a Pitch Class Vector. The DataFrame needs to
            contain at least the columns 'duration_qb' and 'tpc' or 'midi', depending
            on ``pitch_class_format``.
        pitch_class_format : :obj:`str`, optional
            | Defines the type of pitch classes to use for the vectors.
            | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
            | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
            | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
            | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
        normalize : :obj:`bool`, optional
            By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
            the PCV for each group.
        ensure_pitch_classes : :obj:`Iterable`, optional
            By default, pitch classes that don't appear don't appear. Pass a collection of pitch
            classes if you want to ensure their presence even if empty. For example, if
            ``pitch_class_format='pc'`` you could pass ``ensure_columns=range(12)``.


        Returns
        -------
        :obj:`pandas.Series`
        """
        if pitch_class_format in ("tpc", "name"):
            pitch_class_grouper = notes.tpc
        elif pitch_class_format == "pc":
            pitch_class_grouper = (notes.midi % 12).rename("pc")
        elif pitch_class_format == "midi":
            pitch_class_grouper = notes.midi
        else:
            print(
                "pitch_class_format needs to be one of 'tpc', 'name', 'pc', 'midi', not "
                + str(pitch_class_format)
            )
            return pd.DataFrame()
        pcvs = notes.groupby(pitch_class_grouper, dropna=False).duration_qb.sum()
        if pitch_class_format == "name":
            pcvs.index = fifths2name(pcvs.index)
        if normalize:
            pcvs = (pcvs / pcvs.sum()).rename("duration_qb_normalized")
        if ensure_pitch_classes is not None:
            missing = [c for c in ensure_pitch_classes if c not in pcvs.index]
            if len(missing) > 0:
                new_values = pd.Series(pd.NA, index=missing)
                pcvs = pd.concat([pcvs, new_values]).sort_index()
        return pcvs


class ChordSymbolUnigrams(ChordSymbolAnalyzer):
    """Analyzer that returns the counts of chord symbols for each group, ordered by descending
    frequency.
    """

    def __init__(self, once_per_group=False):
        """Analyzer that returns the counts of chord symbols for each group, ordered by descending
            frequency.

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one unigram ranking per group item.
            Set to True to instead compute only one for each group.
        """
        super().__init__(once_per_group=once_per_group)
        self.level_names["processed"] = "chord"

    @staticmethod
    def compute(expanded):
        """Computes the value counts of the chord symbol column.

        Parameters
        ----------
        expanded : :obj:`pandas.DataFrame`
            Expanded harmony labels.

        Returns
        -------
        :obj:`pandas.Series`
            The last index level has unique chord symbols, Series values are their corresponding
            counts.
        """
        if len(expanded) == 0:
            return pd.Series()
        return expanded.chord.value_counts().rename("count")


class ChordSymbolBigrams(ChordSymbolAnalyzer):
    """Analyzer that returns the bigram counts for all valid chord transitions within a group,
    ordered by descending frequency.
    """

    def __init__(self, once_per_group=False):
        """Analyzer that returns the bigram counts for all valid chord transitions within a group,
            ordered by descending frequency.

        Parameters
        ----------
        once_per_group : :obj:`bool`
            By default, computes one bigram ranking per group item.
            Set to True to instead compute only one for each group.
        """
        super().__init__(once_per_group=once_per_group)
        self.level_names["processed"] = ["from", "to"]
        self.group2pandas = "group_of_series2series"

    def check(self, df):
        if df.shape[0] < 2:
            return False, "DataFrame has only one row, cannot compute bigram."
        if df.localkey.nunique() > 1 and df.index.nlevels == 1:
            return (
                False,
                "DataFrame contains labels from several local keys but no MultiIndex.",
            )
        return True, ""

    @staticmethod
    def compute(expanded):
        """Turns the chord column into bigrams and returns their counts in descending order.

        Parameters
        ----------
        expanded : :obj:`pandas.DataFrame`
            Expanded harmony labels.

        Returns
        -------
        :obj:`pandas.Series`
            The last two index level are unique (from, to) bigrams, Series values are their
            corresponding counts.
        """
        if len(expanded) == 0:
            return pd.Series()
        n_index_levels = expanded.index.nlevels
        if n_index_levels > 1:
            # create a nested list to exclude transitions between groups
            index_levels_but_last = list(range(n_index_levels - 1))
            gpb = expanded.groupby(level=index_levels_but_last)
            assert all(gpb.localkey.nunique() == 1), (
                f"Grouping by the first {n_index_levels-1} "
                f"index levels does not result in localkey segments. Has "
                f"the LocalKeySlicer been applied?\n{gpb.localkey.nunique()}"
            )
            chords = gpb.chord.apply(list).to_list()
        else:
            chords = expanded.chords.to_list()
        bigrams = grams(chords, n=2)
        expanded = pd.DataFrame(bigrams)
        try:
            counts = (
                expanded.groupby([0, 1])
                .size()
                .sort_values(ascending=False)
                .rename("count")
            )
        except KeyError:
            print(expanded)
            raise
        return counts
