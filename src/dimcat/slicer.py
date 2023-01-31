"""A Slicer is a PipelineStep that cuts Data into segments, effectively multiplying IDs."""
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from ms3 import (
    overlapping_chunk_per_interval,
    replace_index_by_intervals,
    segment_by_adjacency_groups,
    segment_by_criterion,
    slice_df,
)

from ._typing import PieceID, SliceID
from .data import AnalyzedData, Data, SlicedData
from .pipeline import PipelineStep
from .utils import interval_index2interval


class Slicer(PipelineStep, ABC):
    """
    A Slicer will process a Data object by extracting chunks of rows based on an IntervalIndex,
    which may potentially result in data points to be duplicated or split in two.

    Concretely, it iterates through index groups and, for each Interval to form a slice of a
    particular piece, creates a new index tuple with the Interval appended.

    If created from a facet, the slicer creates a pandas.Series per slice containing metadata for
    later grouping, and stores it in ``data.slice_info[(corpus, fname, Interval)]``.  The
    slices generated from this can be found in ``data.sliced[facet][(corpus, fname, Interval)]``.
    """


class FacetSlicer(Slicer):
    """A FacetSlicer creates slice information based on one particular data facet, e.g.
    note lists or annotations.
    """

    def __init__(self, facet, slice_name, **config):
        self.required_facets = [facet]
        self.level_names = dict(
            indices=slice_name,  # index level name for the slice intervals
            slicer=slice_name + "d",
        )  # file name component for the filename_factory
        self.config = dict(config)
        """Define {"indices": "slice_level_name"} to give the third index column that contains
        the slices' intervals a meaningful name. Define {"slicer": "slicer_name"} for the
        creation of meaningful file names.
        """

    @abstractmethod
    def iter_slices(self, index, facet_df, **config) -> Tuple[tuple, pd.Series]:
        """Generates slice intervals and slice_info Series based on facet_df (e.g. notes or annotations) and
        iterates through them so that they can be stored and then used for slicing the facet.

        Yields
        ------
        :obj:`tuple`
            The DataFrame's index with the slice's quarterbeat interval appended. Will be used to
            store and look up information on this particular slice.
        :obj:`pandas.Series`
            Information about the slice, especially the feature value(s) based on which the segment
            has come about, such as the local key. Mainly used for subsequent grouping of slices.
        """

    def process_data(self, data: Data) -> SlicedData:
        assert not isinstance(
            data, AnalyzedData
        ), "Data object already contains processed data. Cannot slice it post-hoc. Apply Slicers beforehand."
        if isinstance(data, SlicedData):
            previous_slicers = [
                step for step in data.pipeline_steps if isinstance(step, Slicer)
            ]
            raise NotImplementedError(
                f"Data object had already been sliced by {previous_slicers}."
            )
        return SlicedData(data)

    def filename_factory(self):
        return self.level_names["slicer"]


class LazyFacetSlicer(FacetSlicer):
    """A FacetSlicer creates slice information based on one particular data facet, e.g.
    note lists or annotations.

    It is called 'lazy' because it does not perform any actual slicing, which will be up to the Data object to do
    upon request.
    """

    @abstractmethod
    def iter_slices(self, index, facet_df, **config) -> Tuple[tuple, pd.Series]:
        """Generates slice intervals and slice_info Series based on facet_df (e.g. notes or annotations) and
        iterates through them so that they can be stored and then used for slicing the facet.

        Yields
        ------
        :obj:`tuple`
            The DataFrame's index with the slice's quarterbeat interval appended. Will be used to
            store and look up information on this particular slice.
        :obj:`pandas.Series`
            Information about the slice, especially the feature value(s) based on which the segment
            has come about, such as the local key. Mainly used for subsequent grouping of slices.
        """

    def process_data(self, data: Data) -> SlicedData:
        result = super().process_data(data)
        # The three dictionaries that will be added to the resulting Data object,
        # where keys are the new indices. Each piece's (corpus, fname) index tuple will be
        # multiplied according to the number of slices and the new index tuples will be
        # differentiated by the slices' intervals: (corpus, fname, interval)
        slice_indices: Dict[PieceID, List[SliceID]] = defaultdict(list)
        slice_infos: Dict[PieceID, pd.DataFrame] = {}
        """For each slice, the relevant info about it, e.g. for later grouping"""

        facet_in_question = self.required_facets[0]
        for index, facet_df in result.iter_facet(facet_in_question):
            try:
                eligible, message = self.check(facet_df)
            except AttributeError:
                print(result)
                raise
            if not eligible:
                print(f"{index}: {message}")
                continue
            for slice_index, slice_info in self.iter_slices(
                index, facet_df, **self.config
            ):
                slice_infos[slice_index] = slice_info
                slice_indices[index].append(slice_index)
        result.track_pipeline(self, **self.level_names)
        result.slice_info = slice_infos
        result.set_indices(dict(slice_indices))
        return result


class OnePassFacetSlicer(FacetSlicer):
    """These slicers use information already available based on the computation of slice intervals and slice_info.
    Either this computation has left them with chunks already, or the facet does not actually have to be sliced because
    chunks can be created by subselecting it with the computed intervals.
    """

    @abstractmethod
    def iter_slices(
        self, index, facet_df, **config
    ) -> Tuple[tuple, pd.Series, pd.DataFrame]:
        """Slices the facet_df (e.g. notes or annotations) and iterates through the slices
        so that they can be stored.

        Yields
        ------
        :obj:`tuple`
            The DataFrame's index with the slice's quarterbeat interval appended. Will be used to
            store and look up information on this particular slice.
        :obj:`pandas.Series`
            Information about the slice, especially the feature value(s) based on which the segment
            has come about, such as the local key. Mainly used for subsequent grouping of slices.
        :obj:`pandas.DataFrame`
            The slice, i.e. a chunk or subselection of ``facet_df``.
        """

    def process_data(self, data: Data) -> SlicedData:
        result = super().process_data(data)
        # The three dictionaries that will be added to the resulting Data object,
        # where keys are the new indices. Each piece's (corpus, fname) index tuple will be
        # multiplied according to the number of slices and the new index tuples will be
        # differentiated by the slices' intervals: (corpus, fname, interval)
        indices = (
            {}
        )  # {group -> [(index)]}; each list of indices will be at least as long as before
        slice_infos = (
            {}
        )  # for each slice, the relevant info about it, e.g. for later grouping
        sliced = (
            {}
        )  # for each piece, the relevant facet sliced into multiple DataFrames
        for group, dfs in result.iter_facet(self.required_facets[0]):
            new_index_group = []
            for index, facet_df in dfs.items():
                eligible, message = self.check(facet_df)
                if not eligible:
                    print(f"{index}: {message}")
                    continue
                for slice_index, slice_info, chunk in self.iter_slices(
                    index, facet_df, **self.config
                ):
                    new_index_group.append(slice_index)
                    sliced[slice_index] = chunk
                    slice_infos[slice_index] = slice_info
            indices[group] = new_index_group
        result.track_pipeline(self, **self.level_names)
        result.sliced[self.required_facets[0]] = sliced
        result.slice_info = slice_infos
        result.indices = indices
        return result


class NoteSlicer(OnePassFacetSlicer):
    """Slices note tables based on a regular interval size or on every onset."""

    def __init__(self, quarters_per_slice=None):
        """Slices note tables based on a regular interval size or on every onset.

        Parameters
        ----------
        quarters_per_slice : :obj:`float`, optional
            By default, the slices have variable size, from onset to onset. If you pass a value,
            the slices will have that constant size, measured in quarter notes. For example,
            pass 1.0 for all slices to have size 1 quarter.
        """
        if quarters_per_slice is None:
            name = "onset_slice"
        else:
            name = f"{round(float(quarters_per_slice), 1)}q_slice"
        if quarters_per_slice is not None:
            quarters_per_slice = float(quarters_per_slice)
        super().__init__("notes", name, quarters_per_slice=quarters_per_slice)

    def check(self, facet_df):
        if len(facet_df.index) == 0:
            return False, "Empty DataFrame."
        return True, ""

    def iter_slices(self, index, facet_df, quarters_per_slice=None):
        try:
            sliced_df = slice_df(facet_df, quarters_per_slice)
            for interval, chunk in sliced_df.items():
                slice_index = index + (interval,)
                if len(chunk.index) == 0:
                    slice_info = pd.Series(dtype="object")
                else:
                    slice_info = chunk.iloc[0].copy()
                yield slice_index, slice_info, chunk
        except AssertionError:
            print(facet_df)
            raise


class MeasureSlicer(OnePassFacetSlicer):
    """Slices note tables based on a regular interval size or on every onset."""

    def __init__(self, use_measure_numbers=True):
        """Slices note tables based on a regular interval size or on every onset.

        Parameters
        ----------
        use_measure_numbers : :obj:`bool`, optional
            By default, slices are created based on the scores' measure numbers (MN). Pass False
            if you want them to reflect the scores' <Measure> tags (MC), which reflects upbeats,
            split measures, etc.
        """
        super().__init__(
            "measures", "measure_slice", use_measure_numbers=use_measure_numbers
        )

    def check(self, facet_df):
        if len(facet_df) == 0:
            return False, "Empty DataFrame"
        return True, ""

    def iter_slices(self, index, facet_df, use_measure_numbers=True):
        if use_measure_numbers:
            mn_value_counts = facet_df.mn.value_counts()
            if (mn_value_counts == 1).all():
                # each row corresponds to a different measure number and thus to a slice,
                # therefore no grouping is needed
                use_measure_numbers = False
        if use_measure_numbers:
            for mn, mn_group in facet_df.groupby("mn"):
                interval = interval_index2interval(mn_group.index)
                slice_index = index + (interval,)
                yield slice_index, mn_group.iloc[0].copy(), mn_group
        else:
            for interval, slice_info in facet_df.iterrows():
                slice_index = index + (interval,)
                yield slice_index, slice_info, pd.DataFrame(slice_info).T


class ChordFeatureSlicer(OnePassFacetSlicer):
    """Create slices based on a particular feature by grouping adjacent identical values."""

    def __init__(self, feature="chord", na_values="ffill"):
        """Create slices based on a particular feature.

        Parameters
        ----------
        feature : :obj:`str`
            Column name used for creating slices from adjacent identical values. Useful, for example, for creating
            key segments.
        na_values : (:obj:`list` of) :obj:`str` or :obj:`Any`, optional
            | Either pass a list of equal length as ``cols`` or a single value that is passed to
            | :func:`adjacency_groups` for each. Not dealing with NA values will lead to wrongly grouped segments.
            | 'pad', 'ffill' (default) groups NA values with the preceding group
            | 'group' creates individual groups for NA values
            | 'backfill' or 'bfill' groups NA values with the subsequent group
            | Any other value works like 'group', with the difference that the NA groups will be named with this value.
        """
        name = f"{feature}_slice"
        super().__init__("expanded", name, feature=feature, na_values=na_values)

    def check(self, facet_df):
        if len(facet_df) == 0:
            return False, "Empty DataFrame"
        for col in ("duration_qb", self.config["feature"]):
            if col not in facet_df.columns:
                return (
                    False,
                    f"Couldn't compute {self.config['feature']} slices because the annotation table is missing "
                    f"the column '{col}'.",
                )
        return True, ""

    def iter_slices(self, index, facet_df, feature="chord", na_values="ffill"):
        logger_name = "_".join(index).replace(".", "")
        segmented = segment_by_adjacency_groups(
            facet_df, cols=feature, na_values=na_values, logger=logger_name
        )
        for (interval, _), row in segmented.iterrows():
            slice_index = index + (interval,)
            selector = facet_df.index.overlaps(interval)
            yield slice_index, row, facet_df[selector]


class LocalKeySlicer(ChordFeatureSlicer):
    """Slices annotation tables based on adjacency groups of the 'localkey' column."""

    def __init__(self):
        """Slices annotation tables based on adjacency groups of the 'localkey' column."""
        super().__init__(feature="localkey", na_values="group")

    def iter_slices(self, index, facet_df, feature="localkey", na_values="group"):
        name = "_".join(index)
        segmented = segment_by_adjacency_groups(
            facet_df, feature, na_values=na_values, logger=name
        )
        missing_localkey = segmented.localkey.isna()
        if missing_localkey.any():
            if missing_localkey.all():
                print(f"No localkey known for {index}. Skipping.")
                return
            else:
                print(
                    f"{index} has segments with unknown localkey:\n"
                    f"{segmented[missing_localkey]}"
                )
                segmented = segmented[~missing_localkey]

        if "localkey_is_minor" not in segmented.columns:
            segmented["localkey_is_minor"] = segmented.localkey.str.islower()
        for (interval, _), row in segmented.iterrows():
            slice_index = index + (interval,)
            selector = facet_df.index.overlaps(interval)
            yield slice_index, row, facet_df[selector]


class ChordCriterionSlicer(OnePassFacetSlicer):
    """Create slices based on a particular criterion such that each True row starts a new slice.

    The 'expanded' labels are not sliced but subselected. For the special case of phrase labels that may actually
    occur without a new harmony and need slice the table, use PhraseSlicer.
    """

    def __init__(self, column="chord", contains_str=None, warn_na=False):
        """Defines the criteria for starting slices.

        Parameters
        ----------
        column : :obj:`str`, optional
            Name of the column against which the criterion will be tested.
        contains_str : :obj:`str`, optional
            Select all rows where ``column`` contains this string.
        warn_na : :obj:`bool`, optional
            If the boolean mask starts with any number of False, this first group will be missing from the result.
            Set warn_na to True if you want the logger to throw a warning in this case.
        """
        if contains_str is None:
            raise NotImplementedError(
                "Currently 'contains_str' is the only criterion implemented."
            )
        name = f"{column}_criterion_slice"
        super().__init__(
            "expanded", name, column=column, contains_str=contains_str, warn_na=warn_na
        )

    def check(self, facet_df):
        if len(facet_df) == 0:
            return False, "Empty DataFrame"
        for col in ("duration_qb", self.config["column"]):
            if col not in facet_df.columns:
                return (
                    False,
                    f"Couldn't compute {self.config['column']} criterion slices because the annotation table is "
                    f"missing the column '{col}'.",
                )
        return True, ""

    def iter_slices(
        self, index, facet_df, column="chord", contains_str=None, warn_na=False
    ):
        logger_name = "_".join(index).replace(".", "")
        if contains_str is not None:
            boolean_mask = facet_df[column].str.contains(contains_str).fillna(False)
        else:
            return
        segmented = segment_by_criterion(
            facet_df, boolean_mask=boolean_mask, warn_na=warn_na, logger=logger_name
        )
        for interval, row in segmented.iterrows():
            slice_index = index + (interval,)
            selector = facet_df.index.overlaps(interval)
            yield slice_index, row, facet_df[selector]


class SpecialFacetSlicer(LazyFacetSlicer):
    """These facet slicers first gather slice intervals and info and then perform an individual kind of slicing
    afterwards.
    """

    @abstractmethod
    def perform_facet_slicing(self, data: Data) -> Data:
        """This is where the special slicing takes place."""

    def process_data(self, data: Data) -> Data:
        data = super().process_data(data)
        sliced_data = self.perform_facet_slicing(data)
        return sliced_data


class PhraseSlicer(SpecialFacetSlicer):
    """Create slice info from phrase beginnings."""

    SPLIT_REGEX = re.compile(
        r"""
                            ^(?P<label>\.?
                                (?:[a-gA-G](?:b*|\#*)\.)?
                                (?:(?:(?:b*|\#*)(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+\.)?
                                (?:(?:(?:b*|\#*)(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+\[)?
                                (?:b*|\#*)(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none)
                                (?:%|o|\+|M|\+M)?
                                (?:7|65|43|42|2|64|6)?
                                (?:\((?:(?:\+|-|\^|v)?(?:b*|\#*)\d)+\))?
                                (?:/(?:(?:b*|\#*)(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*)?
                                \]?
                             )?
                             (?P<cadence_phrase>
                                (?:\|(?:HC|PAC|IAC|DC|EC|PC)(?:\..+?)?)?
                                (?:\\\\|\}\{|\{|\})?
                             )?
                            $
                            """,
        re.VERBOSE,
    )
    OVERWRITE_FEATURES = [
        "alt_label",
        "chord",
        "numeral",
        "special",
        "form",
        "figbass",
        "changes",
        "relativeroot",
        "chord_type",
        "chord_tones",
        "added_tones",
        "root",
        "bass_note",
    ]

    def __init__(self, warn_na=False):
        """Defines the criteria for starting slices.

        Parameters
        ----------
        warn_na : :obj:`bool`, optional
            If the boolean mask starts with any number of False, this first group will be missing from the result.
            Set warn_na to True if you want the logger to throw a warning in this case.
        """
        name = "phrase_slice"
        super().__init__("expanded", name, warn_na=warn_na)

    def check(self, facet_df):
        if len(facet_df) == 0:
            return False, "Empty DataFrame"
        for col in ("duration_qb", "phraseend"):
            if col not in facet_df.columns:
                return (
                    False,
                    f"Couldn't compute {self.config['column']} criterion slices because the annotation table is "
                    f"missing the column '{col}'.",
                )
        if facet_df["phraseend"].isna().all():
            return False, "No phrase labels present"
        return True, ""

    def iter_slices(self, index, facet_df, warn_na=False):
        logger_name = "_".join(index).replace(".", "")
        boolean_mask = facet_df["phraseend"].str.contains("{").fillna(False)
        if not boolean_mask.any():
            boolean_mask = facet_df["phraseend"].str.contains("////").fillna(False)
        segmented = segment_by_criterion(
            facet_df, boolean_mask=boolean_mask, warn_na=warn_na, logger=logger_name
        )
        for interval, row in segmented.iterrows():
            slice_index = index + (interval,)
            yield slice_index, row

    def perform_facet_slicing(self, data: Data) -> Data:
        data.sliced["expanded"] = {}
        facet_ids = defaultdict(list)
        for corpus, fname, interval in data.slice_info.keys():
            facet_ids[(corpus, fname)].append(interval)
        for ix, intervals in facet_ids.items():
            facet_df = data.get_item(ix, what="expanded")
            if facet_df is None or len(facet_df.index) == 0:
                continue
            # the following call sets this function apart from the usual slicing via Corpus.slice_facet_if_necessary()
            facet_df = split_chord_and_cadence_phrase(facet_df)
            sliced = overlapping_chunk_per_interval(facet_df, intervals)
            # after the usual slicing, cadence labels from the beginnings of chunks need to be moved to the ending
            # of the previous chunk :-(((
            try:
                sliced = move_cadence_labels_between_chunks(sliced)
            except Exception:
                print(ix)
                from IPython.display import display

                display(facet_df)
                raise
            data.sliced["expanded"].update(
                {ix + (iv,): chunk for iv, chunk in sliced.items()}
            )
        if len(data.sliced["expanded"]) == 0:
            del data.sliced["expanded"]
        return data


def split_chord_and_cadence_phrase(expanded: pd.DataFrame) -> pd.DataFrame:
    """Labels containing a harmony label AND cadence or phrase label are separated into two individual rows."""
    matches = expanded.label.str.extract(PhraseSlicer.SPLIT_REGEX).fillna("")
    need_split = (matches != "").all(axis=1)
    new_duplicate_rows = expanded[need_split].copy()
    # new_duplicate_rows.loc[:, 'label'] = matches.cadence_phrase[need_split]
    overwrite = [
        col
        for col in PhraseSlicer.OVERWRITE_FEATURES
        if col in new_duplicate_rows.columns
    ]
    new_duplicate_rows.loc[:, overwrite] = pd.NA
    new_duplicate_rows.loc[:, "duration_qb"] = 0.0
    # expanded.loc[need_split, 'label'] = matches.label[need_split]
    expanded.loc[need_split, ["phraseend", "cadence"]] = pd.NA
    result = pd.concat([expanded, new_duplicate_rows], ignore_index=True).sort_values(
        ["quarterbeats", "duration_qb"]
    )
    return replace_index_by_intervals(result)


def move_cadence_labels_between_chunks(sliced_dict):
    backward_key_iterator1 = reversed(sliced_dict.keys())
    backward_key_iterator2 = reversed(sliced_dict.keys())
    chunk2 = sliced_dict[next(backward_key_iterator1)]
    for iv1, iv2 in zip(backward_key_iterator1, backward_key_iterator2):
        chunk1 = sliced_dict[iv1]
        first_row = chunk2.iloc[0]
        assert first_row.duration_qb == 0, (
            f"First row should be an individual phrase beginning with duration_qb==0.0"
            f" but looks like this {first_row}\n"
        )
        zero_interval = first_row.name
        is_cadence_label = not pd.isnull(first_row.cadence)
        has_phrase_label = not pd.isnull(first_row.phraseend)
        is_phraseend = has_phrase_label and "}" in first_row.phraseend
        if is_cadence_label or is_phraseend:
            new_last_row = chunk2.iloc[1].to_dict()
            assert (
                new_last_row["quarterbeats"] == first_row.quarterbeats
            ), "Second row should be a chord label with the same onset as the phrase beginning."
            if is_phraseend:
                new_last_row["phraseend"] = first_row.phraseend
            if is_cadence_label:
                new_last_row["cadence"] = first_row.cadence
                # remove cadence from first row of chunk2
                cadence_col = chunk2.columns.get_loc("cadence")
                chunk2.iloc[0, cadence_col] = pd.NA
            new_row = pd.DataFrame.from_records([new_last_row], index=[zero_interval])
            chunk1 = pd.concat([chunk1, new_row])
        sliced_dict[iv1] = chunk1
        chunk2 = chunk1
    return sliced_dict
