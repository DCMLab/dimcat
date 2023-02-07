"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Collection,
    Iterable,
    Iterator,
    List,
    Literal,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import dimcat.data as data_module
import pandas as pd
from ms3 import add_weighted_grace_durations, fifths2name

from ._typing import ID
from .data import AnalyzedData, AnalyzedGroupedDataset, AnalyzedSlicedDataset, _Dataset
from .pipeline import _STR2STEP, PipelineStep
from .utils import grams, make_suffix

logger = logging.getLogger(__name__)


def _stepstrings2steps(stepstrings: Union[str, Collection[str]]) -> Tuple[Type, ...]:
    """Turns one or several names of :obj:`PipelineStep` objects into a
    tuple of references to these classes."""
    if isinstance(stepstrings, str):
        stepstrings = [stepstrings]
    result = [_STR2STEP[step] for step in stepstrings]
    return tuple(result)


def _typestring2type(typestring: str) -> Type:
    return getattr(data_module, typestring)


R = TypeVar("R")


class Analyzer(PipelineStep, ABC):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed."""

    result_type = "Result"

    assert_steps: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if at least one of the names does not belong to
    a :obj:`PipelineStep` that is among the previous PipelineSteps applied to the :obj:`_Dataset`."""

    assert_previous_step: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    :obj:`_Dataset` does not match any of these types."""

    excluded_steps: Union[str, Collection[str]] = []
    """Analyzer.process_data() raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    :obj:`_Dataset` matches one of these types."""

    def __init__(self):
        """Creates essential fields."""
        self.config = {}
        """:obj:`dict`
        This dictionary stores the parameters to be passed to the compute() method."""
        self.group2pandas = None
        """:obj:`str`
        The name of the function that allows displaying one group's results as a single
        pandas object. See data.Corpus.convert_group2pandas()"""
        self.level_names = {}
        """:obj:`dict`
        Define {"indices": "index_level_name"} if the analysis is applied once per group,
        because the index of the DataFrame holding the processed data won't be showing the
        individual indices anymore.
        """

    @staticmethod
    @abstractmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    @abstractmethod
    def compute(self, **kwargs) -> R:
        """Static method that performs the actual computation takes place."""

    @abstractmethod
    def data_iterator(self, data: AnalyzedData) -> Iterator[Tuple[ID, Any]]:
        """How a particular analyzer iterates through a dataset, getting the chunks passed to :meth:`compute`."""
        yield from data

    def process_data(self, dataset: _Dataset) -> AnalyzedData:
        """Returns an :obj:`AnalyzedData` copy of the Dataset with the added analysis result."""
        analyzer_name = self.__class__.__name__
        if len(self.assert_steps) > 0:
            assert_steps = _stepstrings2steps(self.assert_steps)
            for step in assert_steps:
                if not any(
                    isinstance(previous_step, step)
                    for previous_step in dataset.pipeline_steps
                ):
                    raise ValueError(
                        f"{analyzer_name} require previous application of a {step.__name__}."
                    )
        if len(self.assert_previous_step) > 0:
            assert_previous_step = _stepstrings2steps(self.assert_previous_step)
            previous_step = dataset.pipeline_steps[0]
            if not isinstance(previous_step, assert_previous_step):
                raise ValueError(
                    f"{analyzer_name} requires the previous pipeline step to be an "
                    f"instance of {self.assert_previous_step}, not {previous_step.__name__}."
                )
        if len(self.excluded_steps) > 0:
            excluded_steps = _stepstrings2steps(self.excluded_steps)
            for step in excluded_steps:
                if any(
                    isinstance(previous_step, step)
                    for previous_step in dataset.pipeline_steps
                ):
                    raise ValueError(
                        f"{analyzer_name} cannot be applied when a {step.__name__} has been applied before."
                    )
        new_dataset = AnalyzedData(dataset)
        result_type = _typestring2type(self.result_type)
        result_object = result_type(
            analyzer=self, dataset_before=dataset, dataset_after=new_dataset
        )
        result_object.config = self.config
        for idx, df in self.data_iterator(new_dataset):
            eligible, message = self.check(df)
            if not eligible:
                logger.info(f"{idx}: {message}")
                continue
            result_object[idx] = self.compute(df, **self.config)
        result_object = self.post_process(result_object)
        new_dataset.track_pipeline(
            self, group2pandas=self.group2pandas, **self.level_names
        )
        new_dataset.set_result(self, result_object)
        return new_dataset

    def post_process(self, processed):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        return processed


class FacetAnalyzer(Analyzer):
    """Analyzers that work on one particular type of DataFrames."""

    @staticmethod
    def aggregate(result_a: pd.Series, result_b: pd.Series) -> pd.Series:
        return result_a.add(result_b, fill_value=0.0)

    def __init__(self):
        """Adds the field :attr:`required_facets`"""
        super().__init__()
        self.required_facets = []

    def data_iterator(self, data: AnalyzedData) -> Iterator[Tuple[ID, pd.DataFrame]]:
        yield from data.iter_facet(self.required_facets[0])


class NotesAnalyzer(FacetAnalyzer):
    def __init__(self):
        """Analyzers that work on notes tables."""
        super().__init__()
        self.required_facets = ["notes"]


class ChordSymbolAnalyzer(FacetAnalyzer):
    def __init__(self):
        """Analyzers that work on expanded annotation tables."""
        super().__init__()
        self.required_facets = ["expanded"]


class TPCrange(NotesAnalyzer):
    """Computes the range from the minimum to the maximum Tonal Pitch Class (TPC)."""

    def __init__(self):
        super().__init__()
        self.level_names["processed"] = "tpc_range"
        self.group2pandas = "group_of_values2series"

    @staticmethod
    def aggregate(result_a: pd.Series, result_b: pd.Series) -> pd.Series:
        lowest = min(result_a.lowest, result_b.lowest)
        highest = max(result_a.highest, result_b.highest)
        result = dict(
            lowest=lowest,
            highest=highest,
            range=highest - lowest,
        )
        return pd.Series(result)

    @staticmethod
    def compute(notes: pd.DataFrame) -> pd.Series:
        """Computes the range from the minimum to the maximum Tonal Pitch Class (TPC).

        Args:
            notes: Notes table with the column ``tpc``

        Returns:
            The difference between the minimal and the maximal tonal pitch class, measured in perfect fifths.
        """
        highest = notes.tpc.max()
        lowest = notes.tpc.min()
        result = dict(lowest=lowest, highest=highest, range=highest - lowest)
        return pd.Series(result)


class PitchClassVectors(NotesAnalyzer):
    """Analyzer that groups notes by their pitch class and aggregates their durations."""

    def __init__(
        self,
        pitch_class_format="tpc",
        weight_grace_durations: float = 0.0,
        normalize=False,
        ensure_pitch_classes=None,
        include_empty=False,
    ):
        """Analyzer that groups notes by their pitch class and aggregates their durations.

        Parameters
        ----------
        pitch_class_format : :obj:`str`, optional
            | Defines the type of pitch classes.
            | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
            | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
            | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
            | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
        weight_grace_durations : :obj:`float`, optional
            By default (0.), grace notes have duration 0. Set this value to include their weighted
            durations in the computation of PCVs, e.g. 0.5.
        normalize : :obj:`bool`, optional
            By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
            the PCV for each slice.
        ensure_pitch_classes : :obj:`Iterable`, optional
            By default, pitch classes that don't appear don't appear. Pass a collection of pitch
            classes if you want to ensure their presence even if empty. For example, if
            ``pitch_class_format='pc'`` you could pass ``ensure_columns=range(12)``.
        include_empty : :obj:`bool`, optional
            By default, indices for which no notes occur will have length 0 PCV Series and will
            therefore not appear in the concatenated results. Set to True if you want to include
            them as empty rows in a post processing step.
        """
        super().__init__()
        self.config = dict(
            pitch_class_format=pitch_class_format,
            weight_grace_durations=float(weight_grace_durations),
            normalize=normalize,
            ensure_pitch_classes=ensure_pitch_classes,
        )
        self.level_names["processed"] = pitch_class_format
        self.group2pandas = "group2dataframe_unstacked"
        self.include_empty = include_empty
        self.used_pitch_classes = set()

    def filename_factory(self):
        """Generate file name component based on the configuration."""
        return make_suffix(
            ("w", self.config["weight_grace_durations"]),
            ("normalized", self.config["normalize"]),
            self.config["pitch_class_format"],
            "pcvs",
        )

    @staticmethod
    def compute(
        notes: pd.DataFrame,
        pitch_class_format: Literal["tpc", "name", "pc", "midi"] = "tpc",
        weight_grace_durations: float = 0.0,
        normalize: bool = False,
        ensure_pitch_classes: Iterable = None,
    ) -> pd.Series:
        """Group notes by their pitch class and aggregate their durations.

        Args:
            notes:
                Note table to be transformed into a Pitch Class Vector. The DataFrame needs to
                contain at least the columns 'duration_qb' and 'tpc' or 'midi', depending
                on ``pitch_class_format``.
            pitch_class_format:
                | Defines the type of pitch classes to use for the vectors.
                | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
                | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
                | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
                | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
            weight_grace_durations:
                By default (0.), grace notes have duration 0. Set this value to include their weighted
                durations in the computation of PCVs, e.g. 0.5.
            normalize:
                By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
                the PCV for each group.
            ensure_pitch_classes:
                By default, pitch classes that don't appear don't appear. Pass a collection of pitch
                classes if you want to ensure their presence even if empty. For example, if
                ``pitch_class_format='pc'`` you could pass ``ensure_columns=range(12)``.

        Returns:
            A pitch class vector where the aggregated duration of each pitch class is given in quarter notes or,
            if ``normalize=True``, as decimal fractions [0.0, 1.0].
        """
        notes = notes.reset_index(drop=True)
        if pitch_class_format in ("tpc", "name"):
            pitch_class_grouper = notes.tpc
        elif pitch_class_format == "pc":
            pitch_class_grouper = (notes.midi % 12).rename("pc")
        elif pitch_class_format == "midi":
            pitch_class_grouper = notes.midi
        else:
            raise ValueError(
                "pitch_class_format needs to be one of 'tpc', 'name', 'pc', 'midi', not "
                + str(pitch_class_format)
            )
        if weight_grace_durations > 0:
            notes = add_weighted_grace_durations(notes, weight=weight_grace_durations)
        try:
            pcvs = notes.groupby(pitch_class_grouper, dropna=False).duration_qb.sum()
        except ValueError:
            print(notes)
            raise
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

    def post_process(self, processed):
        """Inserts empty pitch class vectors for chunks without pitches, if any."""
        if not self.include_empty:
            return processed
        empty_pcv_ixs = []
        for ix, pcv in processed.items():
            if len(pcv) == 0:
                empty_pcv_ixs.append(ix)
            else:
                self.used_pitch_classes.update(pcv.index)
        if len(empty_pcv_ixs) > 0:
            empty_pcv = pd.Series(pd.NA, index=self.used_pitch_classes)
            for ix in empty_pcv_ixs:
                processed[ix] = empty_pcv
        return processed


class ChordSymbolUnigrams(ChordSymbolAnalyzer):
    """Analyzer that returns the counts of chord symbols for each group, ordered by descending
    frequency.
    """

    def __init__(self):
        """Analyzer that returns the counts of chord symbols for each group, ordered by descending
        frequency.


        """
        super().__init__()
        self.level_names["processed"] = "chord"
        self.group2pandas = "group_of_series2series"

    @staticmethod
    def compute(expanded: pd.DataFrame) -> pd.Series:
        """Computes the value counts of the chord symbol column.

        Args:
            expanded: Expanded harmony labels

        Returns:
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

    assert_steps = ["LocalKeySlicer"]

    def __init__(self, dropna=True):
        """Analyzer that returns the bigram counts for all valid chord transitions within a group,
            ordered by descending frequency.

        Parameters
        ----------
        dropna : :obj:`bool`, optional
            By default, NaN values are dropped before computing bigrams, resulting in transitions
            from a missing value's preceding to its subsequent value. Pass False to include
            bigrams from and to NaN values.
        """
        super().__init__()
        self.level_names["processed"] = ["from", "to"]
        self.group2pandas = "group_of_series2series"
        self.config["dropna"] = dropna

    def check(self, df):
        if len(df.index) < 2:
            return False, "DataFrame has only one row, cannot compute bigram."
        if df.localkey.nunique() > 1 and df.index.nlevels == 1:
            return (
                False,
                "DataFrame contains labels from several local keys but no MultiIndex.",
            )
        return True, ""

    @staticmethod
    def compute(expanded: pd.DataFrame, dropna: bool = True) -> pd.Series:
        """Turns the chord column into bigrams and returns their counts in descending order.

        Args:
            expanded: Expanded harmony table with the columns ``localkey`` and ``chord``.
            dropna:
                By default, NaN values are dropped before computing bigrams, resulting in transitions
                from a missing value's preceding to its subsequent value. Pass False to include
                bigrams from and to NaN values.

        Returns:
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
            if dropna:
                chords = gpb.chord.apply(lambda S: S.dropna().to_list()).to_list()
            else:
                chords = gpb.chord.apply(list).to_list()
        else:
            if dropna:
                chords = expanded.chord.dropna().to_list()
            else:
                chords = expanded.chord.to_list()
        bigrams = grams(chords, n=2)
        expanded = pd.DataFrame(bigrams, columns=["from", "to"])
        try:
            counts = (
                expanded.groupby(["from", "to"])
                .size()
                .sort_values(ascending=False)
                .rename("count")
            )
        except KeyError:
            print(expanded)
            raise
        return counts


class SliceInfoAnalyzer(Analyzer):
    """"""

    assert_steps = ["Slicer"]

    def check(self, df):
        if len(df) == 0:
            return False, "DataFrame is empty."
        return True, ""

    def data_iterator(
        self, data: AnalyzedSlicedDataset
    ) -> Iterator[Tuple[Literal["all_slices"], pd.DataFrame]]:
        yield from [("all_slices", data.get_slice_info())]


class GroupedSliceInfoAnalyzer(SliceInfoAnalyzer):
    """"""

    assert_steps = ["Slicer", "Grouper"]

    def data_iterator(
        self, data: AnalyzedGroupedDataset
    ) -> Iterator[Tuple[ID, pd.DataFrame]]:
        yield from data.iter_grouped_slice_info()


class LocalKeySequence(GroupedSliceInfoAnalyzer):
    """"""

    assert_steps = ["LocalKeySlicer", "PieceGrouper"]

    def __init__(
        self,
    ):
        """"""
        super().__init__()
        self.level_names["processed"] = ["localkeys"]
        self.group2pandas = None

    @staticmethod
    def aggregate(result_a: pd.Series, result_b: pd.Series) -> pd.Series:
        return pd.concat(result_a, result_b)

    @staticmethod
    def compute(slice_info: pd.DataFrame) -> pd.Series:
        return slice_info.localkey


class LocalKeyUnique(SliceInfoAnalyzer):
    """"""

    assert_steps = ["LocalKeySlicer"]

    def __init__(
        self,
    ):
        """"""
        super().__init__()
        self.level_names["processed"] = ["localkeys"]
        self.group2pandas = None

    @staticmethod
    def aggregate(result_a: List[str], result_b: List[str]) -> List[str]:
        set_a = set(result_a)
        set_a.update(result_b)
        return sorted(set_a)

    @staticmethod
    def compute(slice_info: pd.DataFrame) -> List[str]:
        return list(slice_info.localkey.unique())
