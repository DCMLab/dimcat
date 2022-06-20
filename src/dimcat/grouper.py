"""A Grouper is a PipelineStep that groups rows of the Data together into subsets."""
from abc import ABC, abstractmethod
from collections import defaultdict

from .data import Data
from .pipeline import PipelineStep
from .slicer import LocalKeySlicer
from .utils import get_composition_year


class Grouper(PipelineStep, ABC):
    """
    A Grouper will process a Data object by iterating through the current index groups and
    introducing subgroups for each. Groupers always work top-down, meaning that they cannot
    group IDs belonging to different groups.

    Concretely, it will turn the ``{(old_groups) -> [(id)]}`` dict stored in data.indices into
    ``{(*old_groups, new_group) -> [(id)]}``.
    """

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = {}
        """Define {"grouper": "index_level_name"} so that the DataFrame index level distinguishing
        the resulting groups has a meaningful name. E.g., for grouping by years, a good index
        level name could be 'year'.
        """

    @abstractmethod
    def criterion(self, index: tuple, data: Data) -> str:
        """Takes one index and (potentially) looks it up in the data object to return the name
        of the new group that the corresponding element is attributed to. The name will be appended
        to the previous group names tuple.
        """

    def process_data(self, data: Data) -> Data:
        """Returns a copy of the Data object where the list of indices for each existing group has
        been further subdivided into smaller groups.
        """
        indices = {}  # this will be the index dictionary of the returned Data object
        for group, index_group in data.iter_groups():
            # Iterate through groups, i.e. a name and a list of index tuples.
            # A group can only be divided into smaller groups or stay the same ("top-down").
            grouped = defaultdict(list)
            for index in index_group:
                # iterate through this group's indices and apply the grouping criterion to each
                new_group = self.criterion(index, data)
                if new_group is None:
                    continue
                grouped[new_group].append(index)
            for new_group, ids in grouped.items():
                indices[group + (new_group,)] = ids
        if self.sort:
            indices = {key: indices[key] for key in sorted(indices.keys())}
        result = data.copy()
        result.track_pipeline(
            self,
            **self.level_names,
        )
        result.indices = indices
        return result


class CorpusGrouper(Grouper):
    """Groups indices that belong to the same corpus."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = dict(grouper="corpus")

    def criterion(self, index: tuple, data: Data) -> str:
        return index[0]


class PieceGrouper(Grouper):
    """Groups indices that belong to the same piece."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = dict(grouper="fname")

    def criterion(self, index: tuple, data: Data) -> str:
        return index[1]


class YearGrouper(Grouper):
    """Groups indices based on the composition years indicated in the metadata."""

    def __init__(self, sort=True):
        """Groups indices together based on the composition year indicated in the metadata."""
        self.sort = sort
        self.level_names = dict(grouper="year")
        self.year_cache = {}

    def criterion(self, index: tuple, data: Data) -> str:
        ix = index[:2]
        if ix in self.year_cache:
            return self.year_cache[ix]
        metadata_dict = data.pieces[ix]["metadata"]
        year = get_composition_year(metadata_dict)
        self.year_cache[ix] = year
        return year

    def process_data(self, data: Data) -> Data:
        result = super().process_data(data=data)
        self.year_cache = {}
        return result


class ModeGrouper(Grouper):
    """Groups indices based on the mode of a given segment. Requires previous application of
    LocalKeySlicer."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = dict(grouper="localkey_is_minor")
        self.slicer = None

    def criterion(self, index: tuple, data: Data) -> str:
        slice_info = data.slice_info[self.slicer][index]
        try:
            return slice_info["localkey_is_minor"]
        except KeyError:
            print(f"Information on localkey not present in the slice_info of {index}:")
            print(slice_info)

    def process_data(self, data: Data) -> Data:
        try:
            self.slicer = next(
                step for step in data.pipeline_steps if isinstance(step, LocalKeySlicer)
            )
        except StopIteration:
            raise Exception(
                f"Previous PipelineSteps do not include LocalKeySlicer: {data.pipeline_steps}"
            )
        return super().process_data(data)
