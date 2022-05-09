"""A Grouper is a PipelineStep that groups rows of the Data together into subsets."""
from abc import ABC
from collections import defaultdict

from .data import Data
from .pipeline import PipelineStep


class Grouper(PipelineStep, ABC):
    """
    A Grouper will process a Data object by iterating through the current index groups and
    introducing subgroups for each. Groupers always work top-down, meaning that they cannot
    group IDs belonging to different groups.

    Concretely, it will turn the ``{(old_groups) -> [(id)]}`` dict stored in data.indices into
    ``{(*old_groups, new_group) -> [(id)]}``.
    """


class CorpusGrouper(Grouper):
    """Groups indices that belong to the same corpus."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort

    def process_data(self, data: Data) -> Data:
        indices = {}
        for group, index_group in data.iter_groups():
            grouped = defaultdict(list)
            for index in index_group:
                grouped[index[0]].append(index[:2])
            for corpus, ids in grouped.items():
                indices[group + (corpus,)] = ids
        if self.sort:
            indices = {key: indices[key] for key in sorted(indices.keys())}
        result = data.copy()
        result.track_pipeline(
            self,
            grouper="corpus",
        )
        result.indices = indices
        return result


def get_composition_year(metadata_dict):
    start = (
        metadata_dict["composed_start"] if "composed_start" in metadata_dict else None
    )
    end = metadata_dict["composed_end"] if "composed_end" in metadata_dict else None
    if start is None and end is None:
        raise "Metadata do not include composition dates."
    if start is None:
        return end
    if end is None:
        return start
    return round((end + start) / 2, ndigits=1)


class YearGrouper(Grouper):
    """Groups indices based on the composition years indicated in the metadata."""

    def __init__(self, sort=True):
        """Groups indices together based on the composition year indicated in the metadata."""
        self.sort = sort

    def process_data(self, data: Data) -> Data:
        indices = {}
        years = {}
        for group, index_group in data.iter_groups():
            grouped = defaultdict(list)
            for index in index_group:
                ix = index[:2]
                if ix in years:
                    year = years[ix]
                else:
                    year = get_composition_year(data.pieces[ix]["metadata"])
                    years[ix] = year
                grouped[year].append(index)
            for year, ids in grouped.items():
                indices[group + (year,)] = ids
        if self.sort:
            indices = {key: indices[key] for key in sorted(indices.keys())}
        result = data.copy()
        result.track_pipeline(
            self,
            grouper="year",
        )
        result.indices = indices
        return result


class ModeGrouper(Grouper):
    """Groups indices based on the mode of a given segment. Requires previous application of
    LocalKeySlicer."""

    def process_data(self, data: Data) -> Data:
        assert "expanded" in data.slice_info, (
            "Couldn't find slice_info. " "Have you applied LocalKeySlicer() before?"
        )
        indices = {}
        for group, index_group in data.iter_groups():
            grouped = defaultdict(list)
            for index in index_group:
                slice_info = data.slice_info["expanded"][index]
                mode = slice_info["localkey_is_minor"]
                grouped[mode].append(index)
            for mode, ids in grouped.items():
                indices[group + (mode,)] = ids
        result = data.copy()
        result.track_pipeline(
            self,
            grouper="is_minor",
        )
        result.indices = indices
        return result
