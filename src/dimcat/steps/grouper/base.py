import logging
from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Sequence

import pandas as pd
from dimcat.base import is_subclass_of
from dimcat.data.dataset.base import Dataset
from dimcat.data.dataset.processed import GroupedDataset
from dimcat.data.resource.dc import DimcatIndex, DimcatResource, PieceIndex
from dimcat.data.resource.features import Feature
from dimcat.steps.base import FeatureProcessingStep
from dimcat.utils import check_name
from marshmallow import fields, pre_load
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Grouper(FeatureProcessingStep):
    new_dataset_type = GroupedDataset
    new_resource_type = None
    output_package_name = None

    class Schema(FeatureProcessingStep.Schema):
        level_name = fields.Str()

    def __init__(self, level_name: str = "grouper", **kwargs):
        super().__init__(**kwargs)
        self._level_name: str = None
        self.level_name = level_name

    @property
    def level_name(self) -> str:
        return self._level_name

    @level_name.setter
    def level_name(self, level_name: str):
        check_name(level_name)
        self._level_name = level_name

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return pd.concat([resource.df], keys=[self.level_name], names=[self.level_name])

    def _make_new_resource(self, resource: Feature) -> Feature:
        """Apply the grouper to a Feature."""
        result_constructor = self._get_new_resource_type(resource)
        results = self.apply_grouper(resource)
        result_name = self.resource_name_factory(resource)
        return result_constructor.from_dataframe(
            df=results,
            resource_name=result_name,
        )

    def _iter_features(self, dataset: Dataset) -> Iterable[DimcatResource]:
        """Iterate over all resources in the dataset's OutputCatalog."""
        return dataset.outputs.iter_resources()

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        new_dataset._pipeline.add_step(self)
        package_name_resource_iterator = self._iter_features(new_dataset)
        processed_resources = defaultdict(list)
        for package_name, resource in package_name_resource_iterator:
            new_resource = self.process_resource(resource)
            processed_resources[package_name].append(new_resource)
        for package_name, resources in processed_resources.items():
            new_package = self._make_new_package(package_name)
            new_package.extend(resources)
            n_processed = len(resources)
            if new_package.n_resources < n_processed:
                if new_package.n_resources == 0:
                    self.logger.warning(
                        f"None of the {n_processed} {package_name} were successfully transformed."
                    )
                else:
                    self.logger.warning(
                        f"Transformation was successful only on {new_package.n_resources} of the "
                        f"{n_processed} features."
                    )
            new_dataset.outputs.replace_package(new_package)
        return new_dataset

    def _post_process_result(self, result: DimcatResource) -> DimcatResource:
        """Change the default_groupby value of the returned Feature."""
        result.update_default_groupby(self.level_name)
        return result


class CustomPieceGrouper(Grouper):
    class Schema(Grouper.Schema):
        grouped_pieces = fields.Nested(DimcatIndex.Schema)

        @pre_load
        def deal_with_dict(self, data, **kwargs):
            if isinstance(data["grouped_pieces"], MutableMapping):
                if "dtype" not in data["grouped_pieces"] or not is_subclass_of(
                    data["grouped_pieces"]["dtype"], DimcatIndex
                ):
                    # dealing with a manually compiled DimcatConfig where grouped_pieces are a grouping dict
                    grouped_pieces = DimcatIndex.from_grouping(data["grouped_pieces"])
                    data["grouped_pieces"] = grouped_pieces.to_dict()
            return data

    @classmethod
    def from_grouping(
        cls,
        piece_groups: Dict[str, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a CustomPieceGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        names: Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouped_pieces = PieceIndex.from_grouping(
            grouping=piece_groups,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=grouped_pieces.names[0], grouped_pieces=grouped_pieces)

    def __init__(
        self,
        level_name: str = "piece_group",
        grouped_pieces: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)
        self._grouped_pieces = None
        self.grouped_pieces = grouped_pieces

    @property
    def grouped_pieces(self) -> DimcatIndex:
        return self._grouped_pieces

    @grouped_pieces.setter
    def grouped_pieces(self, grouped_pieces: DimcatIndex):
        if isinstance(grouped_pieces, pd.Index):
            grouped_pieces = DimcatIndex(grouped_pieces)
        elif isinstance(grouped_pieces, dict):
            raise TypeError(
                f"Use {self.name}.from_dict() to create a {self.name}from xfrom a dictionary."
            )
        elif not isinstance(grouped_pieces, DimcatIndex):
            raise TypeError(f"Expected DimcatIndex, got {type(grouped_pieces)}")
        if grouped_pieces.names[-1] != "piece":
            raise ValueError(
                f"Expected last level to to be named 'piece', not {grouped_pieces.names[-1]}"
            )
        self._grouped_pieces = grouped_pieces

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return resource.align_with_grouping(self.grouped_pieces)
