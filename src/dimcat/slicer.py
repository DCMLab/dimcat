"""A Slicer is a PipelineStep that cuts Data into segments, effectively multiplying IDs."""
from abc import ABC

from ms3 import segment_by_adjacency_groups, slice_df

from .data import Data
from .pipeline import PipelineStep


class Slicer(PipelineStep, ABC):
    """
    A Slicer will process a Data object by extracting chunks of rows based on an IntervalIndex,
    which may potentially result in data points to be duplicated or split in two.

    Concretely, it iterates through index groups and, for each Interval to form a slice of a
    particular piece, creates a new index tuple with the Interval appended.

    If created from a facet, the slicer creates a pandas.Series per slice containing metadata for
    later grouping, and stores it in ``data.slice_info[facet][(corpus, fname, Interval)]``.  The
    slices generated from this can be found in ``data.sliced[facet][(corpus, fname, Interval)]``.
    """


class NoteSlicer(Slicer):
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
        self.required_facets = ["notes"]
        self.quarters_per_slice = quarters_per_slice

    def process_data(self, data: Data) -> Data:
        assert (
            len(data.processed) == 0
        ), "I don't know how to slice the processed data contained."
        sliced = {}
        slice_info = {}
        indices = {}
        for group, dfs in data.iter_facet("notes"):
            new_index_group = []
            for index, notes in dfs.items():
                sliced_df = slice_df(notes, self.quarters_per_slice)
                for interval, slice in sliced_df.groupby(level=0):
                    slice_index = index + (interval,)
                    new_index_group.append(slice_index)
                    sliced[slice_index] = slice
                    slice_info[slice_index] = slice.iloc[0].copy()
            indices[group] = new_index_group
        result = data.copy()
        result.sliced["notes"] = sliced
        result.slice_info["notes"] = slice_info
        result.indices = indices
        return result


class LocalKeySlicer(Slicer):
    """Slices annotation tables based on adjacency groups of the 'localkey' column."""

    def __init__(self):
        """Slices annotation tables based on adjacency groups of the 'localkey' column."""
        self.required_facets = ["expanded"]

    def process_data(self, data: Data) -> Data:
        assert (
            len(data.processed) == 0
        ), "I don't know how to slice the processed data contained."
        sliced = {}
        slice_info = {}
        indices = {}
        for group, dfs in data.iter_facet("expanded"):
            new_index_group = []
            for index, expanded in dfs.items():
                if len(expanded) == 0:
                    continue
                segmented = segment_by_adjacency_groups(expanded, "localkey")
                for (interval, _), row in segmented.iterrows():
                    slice_index = index + (interval,)
                    new_index_group.append(slice_index)
                    slice_info[slice_index] = row
                    selector = expanded.index.overlaps(interval)
                    sliced[slice_index] = expanded[selector]
            indices[group] = new_index_group
        result = data.copy()
        result.sliced["expanded"] = sliced
        result.slice_info["expanded"] = slice_info
        result.indices = indices
        return result
