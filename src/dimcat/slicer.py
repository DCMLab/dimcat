"""A Slicer is a PipelineStep that cuts Data into segments, effectively multiplying IDs."""
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from ms3 import segment_by_adjacency_groups, segment_by_criterion, slice_df

from .data import Data
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

    def process_data(self, data: Data) -> Data:
        assert len(data.processed) == 0, (
            "Data object already contains processed data. Cannot slice it post-hoc, apply slicers "
            "beforehand."
        )
        previous_slicers = [
            step for step in data.pipeline_steps if isinstance(step, Slicer)
        ]
        if len(previous_slicers) > 0:
            raise NotImplementedError(
                f"Data object had already been sliced by {previous_slicers[0]}."
            )
        return data


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

    def process_data(self, data: Data) -> Data:
        data = super().process_data(data)
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
        facet_in_question = self.required_facets[0]
        for group, dfs in data.iter_facet(facet_in_question):
            new_index_group = []
            for index, facet_df in dfs.items():
                eligible, message = self.check(facet_df)
                if not eligible:
                    print(f"{index}: {message}")
                    continue
                for slice_index, slice_info in self.iter_slices(
                    index, facet_df, **self.config
                ):
                    new_index_group.append(slice_index)
                    slice_infos[slice_index] = slice_info
            indices[group] = new_index_group
        result = data.copy()
        result.track_pipeline(self, **self.level_names)
        result.slice_info = slice_infos
        result.indices = indices
        return result

    def filename_factory(self):
        return self.level_names["slicer"]


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

    def process_data(self, data: Data) -> Data:
        data = super().process_data(data)
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
        for group, dfs in data.iter_facet(self.required_facets[0]):
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
        result = data.copy()
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


class PhraseSlicer(LazyFacetSlicer):
    """Create slice info from phrase beginnings."""

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
        return True, ""

    def iter_slices(self, index, facet_df, warn_na=False):
        logger_name = "_".join(index).replace(".", "")
        boolean_mask = facet_df["phraseend"].str.contains("{").fillna(False)
        segmented = segment_by_criterion(
            facet_df, boolean_mask=boolean_mask, warn_na=warn_na, logger=logger_name
        )
        for interval, row in segmented.iterrows():
            slice_index = index + (interval,)
            yield slice_index, row
