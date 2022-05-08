"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from abc import ABC, abstractmethod

import pandas as pd
from ms3 import fifths2name

from .data import Data
from .pipeline import PipelineStep
from .utils import grams


def dict_of_series_result_to_dataframe(result, short_ids=False):
    key = list(result.keys())[0]
    if len(result) == 1 and not isinstance(key[0], str):
        df = pd.DataFrame(result[key]).T
        df.index = [key]
        name = "fnames" if short_ids else "IDs"
        df.index.rename(name, inplace=True)
    else:
        df = pd.concat(result.values(), keys=result.keys()).unstack()
        nlevels = df.index.nlevels
        level_names = (
            ["fname", "interval"] if short_ids else ["corpus", "fname", "interval"]
        )
        if nlevels == 1:
            df.index.rename(level_names[0], inplace=True)
        else:
            df.index.rename(level_names[:nlevels], inplace=True)
    return df


def dict_of_series_result_to_series(result, short_ids=False):
    df = dict_of_series_result_to_dataframe(result, short_ids=short_ids)
    return df.stack()


class FacetAnalyzer(PipelineStep, ABC):
    """Analyzers that work on one particular type of DataFrames."""

    def __init__(self, concat_groups=False):
        """

        Parameters
        ----------
        concat_groups : :obj:`bool`
            By default, computes one result per group.
            Set to False to instead compute one result per group item.
        """
        self.required_facets = []
        self.concat_groups = concat_groups
        self.config = {}

    @abstractmethod
    def compute(self, df):
        """Where the actual computation takes place."""

    def process_data(self, data: Data) -> Data:
        processed = {}
        for group, dfs in data.iter_facet(
            self.required_facets[0], concatenate=self.concat_groups
        ):
            processed[group] = {
                ID: self.compute(df, **self.config) for ID, df in dfs.items()
            }
        result = data.copy()
        result.load_processed(processed)
        return result


class NotesAnalyzer(FacetAnalyzer):
    def __init__(self, concat_groups=False):
        """Analyzers that work on notes tables.

        Parameters
        ----------
        concat_groups : :obj:`bool`
            By default, computes one result per group.
            Set to False to instead compute one result per group item.
        """
        self.required_facets = ["notes"]
        self.concat_groups = concat_groups
        self.config = {}


class ChordSymbolAnalyzer(FacetAnalyzer):
    def __init__(self, concat_groups=False):
        """Analyzers that work on expanded annotation tables.

        Parameters
        ----------
        concat_groups : :obj:`bool`
            By default, computes one result per group.
            Set to False to instead compute one result per group item.
        """
        self.required_facets = ["expanded"]
        self.concat_groups = concat_groups
        self.config = {}


class TPCrange(NotesAnalyzer):
    """Computes the range from the minimum to the maximum Tonal Pitch Class (TPC)."""

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

    def __init__(self, concat_groups=False, pitch_class_format="tpc", normalize=False):
        """Analyzer that groups notes by their pitch class and aggregates their durations.

        Parameters
        ----------
        concat_groups : :obj:`bool`
            By default, computes one result per group.
            Set to False to instead compute one result per group item.
        pitch_class_format : :obj:`str`, optional
            | Defines the type of pitch classes.
            | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
            | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
            | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
            | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
        normalize : :obj:`bool`, optional
            By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
            the PCV for each slice.
        """
        super().__init__(concat_groups=concat_groups)
        self.config = dict(pitch_class_format=pitch_class_format, normalize=normalize)

    @staticmethod
    def compute(
        notes,
        index_levels=None,
        pitch_class_format="tpc",
        normalize=False,
        fillna=True,
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

    def process_data(self, data: Data) -> Data:
        result = super().process_data(data)
        result._result_to_pandas = dict_of_series_result_to_dataframe
        return result


class ChordSymbolUnigrams(ChordSymbolAnalyzer):
    @staticmethod
    def compute(df):
        if len(df) == 0:
            return pd.Series()
        return df.chord.value_counts()

    def process_data(self, data: Data) -> Data:
        result = super().process_data(data)
        result._result_to_pandas = dict_of_series_result_to_series
        return result


class ChordSymbolBigrams(ChordSymbolAnalyzer):
    @staticmethod
    def compute(df):
        if len(df) == 0:
            return pd.Series()
        bigrams = grams(df.chord.values, n=2)
        df = pd.DataFrame(bigrams)
        counts = df.groupby([0, 1]).size().sort_values(ascending=False)
        return counts

    def process_data(self, data: Data) -> Data:
        result = super().process_data(data)
        result._result_to_pandas = dict_of_series_result_to_series
        return result
