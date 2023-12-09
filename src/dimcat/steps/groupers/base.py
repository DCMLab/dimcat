import logging
from collections import defaultdict
from typing import ClassVar, Dict, Hashable, List, MutableMapping, Optional, Sequence

import marshmallow as mm
import pandas as pd
from dimcat import Dataset
from dimcat.base import FriendlyEnumField, is_subclass_of
from dimcat.data.datasets.processed import GroupedDataset
from dimcat.data.resources.base import DR, D, FeatureName
from dimcat.data.resources.dc import (
    DimcatIndex,
    DimcatResource,
    PieceIndex,
    UnitOfAnalysis,
)
from dimcat.dc_exceptions import (
    DatasetNotProcessableError,
    GrouperNotSetUpError,
    ResourceIsMissingCorpusIndexError,
    ResourceIsMissingPieceIndexError,
)
from dimcat.steps.base import ResourceTransformation
from dimcat.utils import check_name
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Grouper(ResourceTransformation):
    # inherited from PipelineStep:
    _new_dataset_type = GroupedDataset
    _new_resource_type = None  # same as input
    _applicable_to_empty_datasets = True
    # inherited from FeatureProcessingStep:
    _allowed_features = None  # any
    _output_package_name = None  # transform 'features'
    _requires_at_least_one_feature = False

    class Schema(ResourceTransformation.Schema):
        level_name = mm.fields.Str()

    def __init__(self, level_name: str = "group", **kwargs):
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

    def _make_new_resource(self, resource: DR) -> DR:
        if self.level_name in resource.get_level_names():
            return self._reorder_levels(resource)
        return super()._make_new_resource(resource)

    def _post_process_result(
        self,
        result: DR,
        original_resource: DimcatResource,
    ) -> DR:
        """Change the default_groupby value of the returned Feature. Subclasses with a
        :obj:`_FilterMixin` also call the filter's :meth:`_post_process_result` method.
        """
        result.update_default_groupby(self.level_name)
        return result

    def _reorder_levels(self, resource: DR) -> DR:
        """Make this grouper's level name the first (if it existed before)."""
        level_names = resource.get_level_names()
        if level_names[0] == self.level_name:
            self.logger.info(
                f"The resource {resource.resource_name} had already been grouped by a {self.name} "
                f"and is returned as is."
            )
            return resource
        else:
            level_names.remove(self.level_name)
            level_names = [self.level_name] + level_names
            new_df = resource.df.reorder_levels(level_names)
            return resource.__class__.from_resource_and_dataframe(
                resource=resource, df=new_df
            )

    def transform_resource(self, resource: DimcatResource) -> D:
        """Apply the grouper to a Feature."""
        return pd.concat([resource.df], keys=[self.level_name], names=[self.level_name])


class _IdGrouper(Grouper):
    """Superclass for CorpusGrouper and PieceGrouper, which both concern the two index levels that are expected to be
    present in all Facets and Features.
    """

    def _process_resource(self, resource: DR) -> DR:
        """Apply this PipelineStep to a :class:`Resource` and return a copy containing the output(s)."""
        resource = self._pre_process_resource(resource)
        level_names = resource.get_level_names()
        if level_names[0] != self.level_name:
            result = self._make_new_resource(resource)
        else:
            result = resource.copy()
        return self._post_process_result(result, resource)


class CorpusGrouper(_IdGrouper):
    """Results will be grouped by the 'corpus' part of the ('corpus', 'piece') index."""

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if self.level_name not in resource.get_level_names():
            raise ResourceIsMissingCorpusIndexError(
                resource.resource_name, self.level_name
            )

    def __init__(self, level_name: str = "corpus", **kwargs):
        super().__init__(level_name=level_name, **kwargs)


class PieceGrouper(_IdGrouper):
    """Results will be grouped by the 'piece' part of the ('corpus', 'piece') index."""

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if self.level_name not in resource.get_level_names():
            raise ResourceIsMissingPieceIndexError(
                resource.resource_name, self.level_name
            )

    def __init__(self, level_name: str = "piece", **kwargs):
        super().__init__(level_name=level_name, **kwargs)


class GroupedUnitsField(mm.fields.Nested):
    def __init__(
        self,
        nested=DimcatIndex.Schema,
        **kwargs,
    ):
        super().__init__(
            nested=nested,
            **kwargs,
        )

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, DimcatIndex):
            return value
        return super()._deserialize(value, attr, data, **kwargs)


class MappingGrouper(Grouper):
    """Superclass for all Groupers that function on the basis of a {group_name: [index_tuples]} dictionary."""

    class Schema(Grouper.Schema):
        grouped_units = GroupedUnitsField(allow_none=True)

        @mm.pre_load
        def deal_with_dict(self, data, **kwargs):
            if (grouped_units := data.get("grouped_units")) and isinstance(
                grouped_units, MutableMapping
            ):
                if "dtype" not in grouped_units or not is_subclass_of(
                    grouped_units["dtype"], DimcatIndex
                ):
                    # dealing with a manually compiled DimcatConfig where grouped_units are a grouping dict
                    grouped_units = DimcatIndex.from_grouping(grouped_units)
                    data["grouped_units"] = grouped_units.to_dict()
            return data

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[Hashable, List[tuple]],
        level_names: Sequence[str] = ("group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a CustomPieceGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouped_units = DimcatIndex.from_grouping(
            grouping=grouping,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=grouped_units.names[0], grouped_units=grouped_units)

    def __init__(
        self,
        level_name: str = "group",
        grouped_units: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)
        self._grouped_units: Optional[DimcatIndex] = None
        if grouped_units is not None:
            self.grouped_units = grouped_units

    @property
    def grouped_units(self) -> DimcatIndex:
        if self._grouped_units is None:
            return DimcatIndex.from_tuples([], (self.level_name, "corpus", "piece"))
        return self._grouped_units

    @grouped_units.setter
    def grouped_units(self, grouped_units: DimcatIndex):
        if isinstance(grouped_units, pd.Index):
            grouped_units = DimcatIndex(grouped_units)
        elif isinstance(grouped_units, dict):
            raise TypeError(
                f"Use {self.name}.from_dict() to create a {self.name}from a dictionary."
            )
        elif not isinstance(grouped_units, DimcatIndex):
            raise TypeError(f"Expected DimcatIndex, got {type(grouped_units)}")
        if grouped_units.names[-1] != "piece":
            raise ValueError(
                f"Expected last level to to be named 'piece', not {grouped_units.names[-1]}"
            )
        self._grouped_units = grouped_units

    def check_resource(self, resource: DimcatResource) -> None:
        if len(self.grouped_units) == 0:
            raise GrouperNotSetUpError(self.dtype)
        super().check_resource(resource)

    def transform_resource(self, resource: DimcatResource) -> D:
        """Apply the grouper to a Feature."""
        return resource.align_with_grouping(self.grouped_units)


class CriterionGrouper(MappingGrouper):
    """Groupers that are fitted to a Dataset by applying their :meth:`criterion` method to the units of analysis
    for a particular resource and grouping the chunks according to the method's outputs.
    """

    _required_feature: ClassVar[FeatureName]
    """Required for CriterionGroupers, the type of Feature that needs to be present in a dataset to fit this grouper."""

    @staticmethod
    def compute_criterion(unit: D) -> Hashable:
        raise NotImplementedError("Please use a subclass of CriterionGrouper.")

    class Schema(MappingGrouper.Schema):
        smallest_unit = FriendlyEnumField(UnitOfAnalysis, metadata={"expose": False})

    def __init__(
        self,
        level_name: str = "criterion",
        grouped_units: DimcatIndex | pd.MultiIndex = None,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        **kwargs,
    ):
        super().__init__(level_name=level_name, grouped_units=grouped_units, **kwargs)
        self._smallest_unit: UnitOfAnalysis = None
        self.smallest_unit = smallest_unit

    @property
    def required_feature(self) -> FeatureName:
        if not self._required_feature:
            raise NotImplementedError(f"Please use a subclass of {self.name}.")
        return self._required_feature

    @property
    def smallest_unit(self) -> UnitOfAnalysis:
        return self._smallest_unit

    @smallest_unit.setter
    def smallest_unit(self, smallest_unit: UnitOfAnalysis):
        if not isinstance(smallest_unit, UnitOfAnalysis):
            smallest_unit = UnitOfAnalysis(smallest_unit)
        self._smallest_unit = smallest_unit

    def check_dataset(self, dataset: Dataset) -> None:
        super().check_dataset(dataset)
        if self.required_feature not in dataset.extractable_features:
            raise DatasetNotProcessableError(self.required_feature)

    def fit_to_dataset(self, dataset: Dataset) -> None:
        feature = dataset.get_feature(self.required_feature)
        feature_df = feature.df
        groupby = feature.get_grouping_levels(self.smallest_unit)
        self.logger.debug(
            f"Using the {feature.resource_name}'s grouping levels {groupby!r}"
        )
        grouping = defaultdict(list)
        grouped_units = defaultdict(list)
        for unit_name, unit in feature_df.groupby(groupby):
            group = self.compute_criterion(unit)
            grouping[group].append(unit_name)
            grouped_units[group].append(unit)
        grouped_units = {
            group: pd.concat(units) for group, units in grouped_units.items()
        }
        grouped_df = pd.concat(grouped_units, names=[self.level_name])
        feature_name = self.resource_name_factory(feature)
        grouped_feature = feature.__class__.from_resource_and_dataframe(
            resource=feature, df=grouped_df, resource_name=feature_name
        )
        features_package = dataset.outputs.get_package_by_name("features")
        features_package.replace_resource(grouped_feature, feature_name)
        group_index = DimcatIndex.from_grouping(grouping, [self.level_name] + groupby)
        if len(self.grouped_units) > 0:
            self.logger.info(f"Replacing existing grouping with {group_index}")
        self.grouped_units = group_index


class CustomPieceGrouper(MappingGrouper):
    """Allows for grouping by specifying a {group_name: [piece_index_tuples]} dictionary."""

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[Hashable, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a CustomPieceGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouped_units = PieceIndex.from_grouping(
            grouping=grouping,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=grouped_units.names[0], grouped_units=grouped_units)

    def __init__(
        self,
        level_name: str = "piece_group",
        grouped_units: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, grouped_units=grouped_units, **kwargs)

    @property
    def grouped_units(self) -> PieceIndex:
        if self._grouped_units is None:
            return PieceIndex.from_tuples([], (self.level_name, "corpus", "piece"))
        return self._grouped_units

    @grouped_units.setter
    def grouped_units(self, grouped_units: PieceIndex):
        if isinstance(grouped_units, pd.Index):
            grouped_units = PieceIndex(grouped_units)
        elif isinstance(grouped_units, dict):
            raise TypeError(
                f"Use {self.name}.from_dict() to create a {self.name}from a dictionary."
            )
        elif not isinstance(grouped_units, PieceIndex):
            if isinstance(grouped_units, DimcatIndex):
                grouped_units = PieceIndex(grouped_units)
            else:
                raise TypeError(f"Expected PieceIndex, got {type(grouped_units)}")
        if grouped_units.names[-1] != "piece":
            raise ValueError(
                f"Expected last level to to be named 'piece', not {grouped_units.names[-1]}"
            )
        self._grouped_units = grouped_units
