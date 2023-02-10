"""A Grouper is a PipelineStep that groups rows of the Data together into subsets."""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from dimcat.data import AnalyzedData, GroupedData, _Dataset
from dimcat.base import PipelineStep
from dimcat.slicer import LocalKeySlicer
from dimcat.utils import get_composition_year

logger = logging.getLogger(__name__)


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
        self.level_names = dict(grouper="name")

    def filename_factory(self):
        return self.level_names["grouper"] + "_wise"

    @abstractmethod
    def criterion(self, index: tuple, dataset: _Dataset) -> str:
        """Takes one index and (potentially) looks it up in the data object to return the name
        of the new group that the corresponding element is attributed to. The name will be appended
        to the previous group names tuple.
        """

    def process_data(self, dataset: _Dataset) -> GroupedData:
        """Returns a copy of the Data object where the list of indices for each existing group has
        been further subdivided into smaller groups.
        """
        new_dataset = GroupedData(dataset)
        if isinstance(dataset, AnalyzedData):
            for result in new_dataset.processed:
                result.dataset_after = new_dataset
        grouped_indices = defaultdict(list)
        for group, index_group in new_dataset.iter_grouped_indices():
            # Iterate through groups, i.e. a name and a list of index tuples.
            # A group can only be divided into smaller groups or stay the same ("top-down").
            for index in index_group:
                # iterate through this group's indices and apply the grouping criterion to each
                new_group = self.criterion(index, new_dataset)
                if new_group is None:
                    logger.info(
                        f"Grouping criterion could not be computed for {index}."
                    )
                    continue
                grouped_indices[group + (new_group,)].append(index)
        if self.sort:
            grouped_indices = {
                key: grouped_indices[key] for key in sorted(grouped_indices.keys())
            }
        new_dataset.track_pipeline(
            self,
            **self.level_names,
        )
        new_dataset.set_grouped_indices(grouped_indices)
        return new_dataset


class CorpusGrouper(Grouper):
    """Groups indices that belong to the same corpus."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = dict(grouper="corpus")

    def criterion(self, index: tuple, dataset: _Dataset) -> str:
        return index[0]


class PieceGrouper(Grouper):
    """Groups indices that belong to the same piece."""

    def __init__(self, sort=True):
        """Groups indices together that belong to the same corpus."""
        self.sort = sort
        self.level_names = dict(grouper="fname")

    def filename_factory(self):
        return "piece_wise"

    def criterion(self, index: tuple, dataset: _Dataset) -> str:
        return index[1]


class YearGrouper(Grouper):
    """Groups indices based on the composition years indicated in the metadata."""

    def __init__(self, sort=True):
        """Groups indices together based on the composition year indicated in the metadata."""
        self.sort = sort
        self.level_names = dict(grouper="year")
        self.year_cache = {}

    def criterion(self, index: tuple, dataset: _Dataset) -> str:
        ix = index[:2]
        if ix in self.year_cache:
            return self.year_cache[ix]
        metadata_dict = dataset.pieces[ix].tsv_metadata
        year = get_composition_year(metadata_dict)
        self.year_cache[ix] = year
        return year

    def process_data(self, dataset: _Dataset) -> GroupedData:
        result = super().process_data(dataset=dataset)
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

    def filename_factory(self):
        return "mode_wise"

    def criterion(self, index: tuple, dataset: _Dataset) -> str:
        slice_info = dataset.slice_info[index]
        try:
            return slice_info["localkey_is_minor"]
        except KeyError:
            print(f"Information on localkey not present in the slice_info of {index}:")
            print(slice_info)

    def process_data(self, dataset: _Dataset) -> GroupedData:
        self.slicer = dataset.get_previous_pipeline_step(of_type=LocalKeySlicer)
        return super().process_data(dataset)
