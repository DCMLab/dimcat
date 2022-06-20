"""A Slicer is a PipelineStep that cuts Data into segments, effectively multiplying IDs."""
from abc import ABC, abstractmethod

from ms3 import segment_by_adjacency_groups, slice_df

from .data import Data
from .pipeline import PipelineStep
from .utils import make_suffix


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


class FacetSlicer(Slicer):
    """A FacetSlicer creates slice information based on one particular data facet, e.g.
    note lists or annotations.
    """

    def __init__(self):
        self.required_facets = []
        self.level_names = {}
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
                for slice_index, slice, slice_info in self.iter_slices(index, facet_df):
                    new_index_group.append(slice_index)
                    sliced[slice_index] = slice
                    slice_infos[slice_index] = slice_info
            indices[group] = new_index_group
        result = data.copy()
        result.track_pipeline(self, **self.level_names)
        result.sliced[self.required_facets[0]] = sliced
        result.slice_info[self] = slice_infos
        result.indices = indices
        return result


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
        self.required_facets = ["notes"]
        self.quarters_per_slice = quarters_per_slice
        if quarters_per_slice is None:
            name = "slice"
        else:
            name = make_suffix(("q", quarters_per_slice)) + "-slice"
        self.level_names = {"indices": name, "slicer": name}

    def iter_slices(self, index, facet_df):
        sliced_df = slice_df(facet_df, self.quarters_per_slice)
        for interval, slice in sliced_df.groupby(level=0):
            slice_index = index + (interval,)
            yield slice_index, slice, slice.iloc[0].copy()


class LocalKeySlicer(FacetSlicer):
    """Slices annotation tables based on adjacency groups of the 'localkey' column."""

    def __init__(self):
        """Slices annotation tables based on adjacency groups of the 'localkey' column."""
        self.required_facets = ["expanded"]
        self.level_names = {"indices": "localkey_slice", "slicer": "localkey"}

    def check(self, facet_df):
        if len(facet_df) == 0:
            return False, "Empty DataFrame"
        if "duration_qb" not in facet_df.columns:
            return (
                False,
                "Couldn't compute localkey slices because annotation table is missing "
                "the column 'duration_qb'.",
            )
        return True, ""

    def iter_slices(self, index, facet_df):
        name = "_".join(index)
        segmented = segment_by_adjacency_groups(facet_df, "localkey", logger=name)
        missing_localkey = segmented.localkey.isna()
        if missing_localkey.any():
            if (~missing_localkey).any():
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
