"""This module contains subclasses of Dataset. They reflect a particular processing status in terms of the previously
applied Slicers, Groupers, and Analyzers. Each of them yields a copied Dataset object exposing additional methods,
which are defined in the relevant mixin classes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from dimcat.base import DimcatConfig
from dimcat.data.resources.base import Rs
from dimcat.dc_exceptions import NoMatchingResourceFoundError

from .base import Dataset

if TYPE_CHECKING:
    from dimcat.data.resources import Result

logger = logging.getLogger(__name__)


class _ProcessedMixin:
    """Base class for the mixin classes used to stitch together the various Dataset subclasses."""

    pass


class _SlicedMixin(_ProcessedMixin):
    pass


class _GroupedMixin(_ProcessedMixin):
    pass


class _AnalyzedMixin(_ProcessedMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.outputs.has_package("results"):
            self.outputs.make_new_package(package_name="results")

    def add_result(self, result: Result):
        """Adds a result to the outputs catalog."""
        self.add_output(resource=result, package_name="results")

    def get_result(self, regex: Optional[str] = None):
        """Returns the last result that matches the given regex or, if None, the last result added."""
        results = self.outputs.get_package("results")
        if regex is None:
            return results.get_resource_by_name()
        results = self.get_results_by_regex(regex=regex)
        if not results:
            raise NoMatchingResourceFoundError(regex, results.package_name)
        else:
            return results[-1]

    def get_result_by_config(self, config: DimcatConfig) -> Rs:
        """Returns the result of the previously applied analyzer with the given name."""
        results = self.outputs.get_package("results")
        return results.get_resource_by_config(config=config)

    def get_result_by_name(self, name: str) -> Rs:
        """Returns the result of the previously applied analyzer with the given name."""
        results = self.outputs.get_package("results")
        return results.get_resource_by_name(name=name)

    def get_results_by_regex(self, regex: str) -> List[Rs]:
        """Returns the result of the previously applied analyzer with the given name."""
        results = self.outputs.get_package("results")
        return results.get_resources_by_regex(regex=regex)

    def get_results_by_type(self, resource_type: type) -> List[Rs]:
        """Returns the result of the previously applied analyzer with the given name."""
        results = self.outputs.get_package("results")
        return results.get_resources_by_type(resource_type=resource_type)


class SlicedGroupedAnalyzedDataset(
    _SlicedMixin, _GroupedMixin, _AnalyzedMixin, Dataset
):
    """A Dataset subclass that has been sliced, grouped, and analyzed."""

    pass


class SlicedGroupedDataset(_SlicedMixin, _GroupedMixin, Dataset):
    """A Dataset subclass that has been sliced and grouped."""

    pass


class SlicedAnalyzedDataset(_SlicedMixin, _AnalyzedMixin, Dataset):
    """A Dataset subclass that has been sliced and analyzed."""

    pass


class GroupedAnalyzedDataset(_GroupedMixin, _AnalyzedMixin, Dataset):
    """A Dataset subclass that has been grouped and analyzed."""

    pass


class SlicedDataset(_SlicedMixin, Dataset):
    """A Dataset subclass that has been sliced."""

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        """Create a new SlicedDataset from a Dataset object."""
        if isinstance(dataset, _GroupedMixin):
            if isinstance(dataset, _AnalyzedMixin):
                return SlicedGroupedAnalyzedDataset.from_dataset(dataset, **kwargs)
            else:
                return SlicedGroupedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, _AnalyzedMixin):
            return SlicedAnalyzedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, Dataset):
            return super().from_dataset(dataset, **kwargs)


class GroupedDataset(_GroupedMixin, Dataset):
    """A Dataset subclass that has been grouped."""

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        """Create a new GroupedDataset from a Dataset object."""
        if isinstance(dataset, _SlicedMixin):
            if isinstance(dataset, _AnalyzedMixin):
                return SlicedGroupedAnalyzedDataset.from_dataset(dataset, **kwargs)
            else:
                return SlicedGroupedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, _AnalyzedMixin):
            return GroupedAnalyzedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, Dataset):
            return super().from_dataset(dataset, **kwargs)


class AnalyzedDataset(_AnalyzedMixin, Dataset):
    """A Dataset subclass that has been analyzed."""

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        """Create a new AnalyzedDataset from a Dataset object."""
        if isinstance(dataset, _GroupedMixin):
            if isinstance(dataset, _SlicedMixin):
                return SlicedGroupedAnalyzedDataset.from_dataset(dataset, **kwargs)
            else:
                return GroupedAnalyzedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, _SlicedMixin):
            return SlicedAnalyzedDataset.from_dataset(dataset, **kwargs)
        elif isinstance(dataset, Dataset):
            return super().from_dataset(dataset, **kwargs)


SlicedDataset.register(SlicedGroupedDataset)
SlicedDataset.register(SlicedAnalyzedDataset)
SlicedDataset.register(SlicedGroupedAnalyzedDataset)
GroupedDataset.register(SlicedGroupedDataset)
GroupedDataset.register(GroupedAnalyzedDataset)
GroupedDataset.register(SlicedGroupedAnalyzedDataset)
AnalyzedDataset.register(SlicedAnalyzedDataset)
AnalyzedDataset.register(GroupedAnalyzedDataset)
AnalyzedDataset.register(SlicedGroupedAnalyzedDataset)
