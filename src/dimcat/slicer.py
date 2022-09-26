"""A Slicer is a PipelineStep that cuts Data into segments, effectively multiplying IDs."""
from abc import ABC, abstractmethod

import pandas as pd
from ms3 import segment_by_adjacency_groups, slice_df

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

    def __init__(self):
        self.required_facets = []
        self.level_names = dict(slicer="name")
        self.config = {}
        """Define {"indices": "slice_level_name"} to give the third index column that contains
        the slices' intervals a meaningful name. Define {"slicer": "slicer_name"} for the
        creation of meaningful file names.
        """

    @abstractmethod
    def iter_slices(self, index, facet_df):
        """Slices the facet_df (e.g. notes or annotations) and iterates through the slices
        so that they can be stored.

        Yields
        ------
        :obj:`tuple`
            The DataFrame's index with the slice's quarterbeat interval appended. Will be used to
            store and look up information on this particular slice.
        :obj:`pandas.DataFrame`
            The slice, i.e. a segment of ``facet_df``.
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
        # The three dictionaries that will be added to the resulting Data object,
        # where keys are the new indices. Each piece's (corpus, fname) index tuple will be
        # multiplied according to the number of slices and the new index tuples will be
        # differentiated by the slices' intervals: (corpus, fname, interval)
        sliced = (
            {}
        )  # for each piece, the relevant facet sliced into multiple DataFrames
        slice_infos = (
            {}
        )  # for each slice, the relevant info about it, e.g. for later grouping
        indices = (
            {}
        )  # {group -> [(index)]}; each list of indices will be at least as long as before
        for group, dfs in data.iter_facet(self.required_facets[0]):
            new_index_group = []
            for index, facet_df in dfs.items():
                eligible, message = self.check(facet_df)
                if not eligible:
                    print(f"{index}: {message}")
                    continue
                for slice_index, chunk, slice_info in self.iter_slices(
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

    def filename_factory(self):
        return self.level_names["slicer"] + "d"


class NoteSlicer(FacetSlicer):
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
        super().__init__()
        self.required_facets = ["notes"]
        if quarters_per_slice is None:
            name = "slice"
        else:
            name = f"{round(float(quarters_per_slice), 1)}q_slice"
        self.level_names = {"indices": name, "slicer": name}
        self.config["quarters_per_slice"] = (
            quarters_per_slice
            if quarters_per_slice is None
            else float(quarters_per_slice)
        )

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
                yield slice_index, chunk, slice_info
        except AssertionError:
            print(facet_df)
            raise


class MeasureSlicer(FacetSlicer):
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
        super().__init__()
        self.required_facets = ["measures"]
        name = "measure_slice"
        self.level_names = {"indices": name, "slicer": name}
        self.config["use_measure_numbers"] = use_measure_numbers

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
                yield slice_index, mn_group, mn_group.iloc[0].copy()
        else:
            for interval, slice_info in facet_df.iterrows():
                slice_index = index + (interval,)
                yield slice_index, pd.DataFrame(slice_info).T, slice_info


class ChordFeatureSlicer(FacetSlicer):
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
        super().__init__()
        self.required_facets = ["expanded"]
        self.level_names = {"indices": f"{feature}_slice", "slicer": feature}
        self.config.update(
            dict(
                feature=feature,
                na_values=na_values,
            )
        )

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
            yield slice_index, facet_df[selector], row


class LocalKeySlicer(ChordFeatureSlicer):
    """Slices annotation tables based on adjacency groups of the 'localkey' column."""

    def __init__(self):
        """Slices annotation tables based on adjacency groups of the 'localkey' column."""
        super().__init__(feature="localkey", na_values="group")

    def iter_slices(self, index, facet_df, feature="localkey", na_values="group"):
        name = "_".join(index)
        segmented = segment_by_adjacency_groups(facet_df, feature, logger=name)
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
            yield slice_index, facet_df[selector], row
