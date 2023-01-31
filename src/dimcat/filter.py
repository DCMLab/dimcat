"""A Filter is a PipelineStep that removes indices from the Data."""
from abc import ABC, abstractmethod

from .data import Data
from .pipeline import PipelineStep


class Filter(PipelineStep, ABC):
    """
    A Filter will process a Data object by removing indices from its list of indices.
    """

    def __init__(self, keep_empty_groups=False):
        self.keep_empty_groups = keep_empty_groups

    @abstractmethod
    def criterion(self, index: tuple, data: Data) -> bool:
        """Takes one index and (potentially) looks it up in the data object to decide whether to
        keep it (returns True) or filter it out (returns False).
        """

    def process_data(self, data: Data) -> Data:
        """Returns a copy of the Data object where the list of indices for each existing group has
        potentially fewer elements than before.
        """
        indices = {}
        for group, index_group in data.iter_grouped_indices():
            new_group = []
            for index in index_group:
                if self.criterion(index, data):
                    new_group.append(index)
            if self.keep_empty_groups or len(new_group) > 0:
                indices[group] = new_group
        result = data.copy()
        result.track_pipeline(
            self,
        )
        result.set_indices(indices)
        return result


class IsAnnotatedFilter(Filter):
    """Keeps only pieces for which an 'expanded' DataFrame is available."""

    def criterion(self, index, data: Data) -> bool:
        corpus, fname, *_ = index
        file, df = data.data[corpus][fname].get_facet("expanded")
        # TODO: logger.debug(file)
        return df is not None and len(df.index) > 0


class HasCadenceAnnotationsFilter(Filter):
    """Keeps only pieces whose 'expanded' includes at least one cadence."""

    def criterion(self, index, data: Data) -> bool:
        corpus, fname, *_ = index
        file, df = data.data[corpus][fname].get_facet("expanded")
        # TODO: logger.debug(file)
        if df is not None and len(df.index) > 0:
            return df.cadence.notna().any()
        return False
