import logging
from typing import Dict, Hashable, List, Sequence

import pandas as pd
from dimcat import Dataset
from dimcat.data.resources import DimcatIndex
from dimcat.data.resources.base import IX, D, FeatureName
from dimcat.data.resources.dc import UnitOfAnalysis
from dimcat.steps.groupers.base import CriterionGrouper, CustomPieceGrouper
from typing_extensions import Self

logger = logging.getLogger(__name__)


class HasCadenceAnnotationsGrouper(CriterionGrouper):
    """Boolean grouper that categorizes slices, pieces, or groups by whether they have at least one cadence label or
    not."""

    _required_feature = FeatureName.DcmlAnnotations

    @staticmethod
    def compute_criterion(unit: D) -> Hashable:
        """Returns True if the unit has a column called 'cadence' containing at least one non-null value."""
        return (unit["cadence"].fillna("") != "").any()

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[bool, List[tuple]],
        level_names: Sequence[str] = ("has_cadence_annotations", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a HasCadenceAnnotations grouper from a dictionary of piece groups. Keys should be True and False.

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
        level_name: str = "has_cadence_annotations",
        grouped_units: DimcatIndex | pd.MultiIndex = None,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        **kwargs,
    ):
        super().__init__(
            level_name=level_name,
            grouped_units=grouped_units,
            smallest_unit=smallest_unit,
            **kwargs,
        )


class HasHarmonyLabelsGrouper(CustomPieceGrouper):
    """Boolean grouper that categorizes pieces by whether they have at least one DCML harmony label or not."""

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[bool, List[tuple]],
        level_names: Sequence[str] = ("has_harmony_labels", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a HasHarmonyLabels grouper from a dictionary of piece groups. Keys should be True and False.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        return super().from_grouping(
            grouping=grouping,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )

    def __init__(
        self,
        level_name: str = "has_harmony_labels",
        grouped_units: IX = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, grouped_units=grouped_units, **kwargs)

    def fit_to_dataset(self, dataset: Dataset) -> None:
        metadata = dataset.get_metadata(raw=True)
        has_labels = metadata.df["label_count"] > 0
        grouping = has_labels.groupby(has_labels, sort=True).groups
        group_index = DimcatIndex.from_grouping(
            grouping, ("has_harmony_labels", "corpus", "piece")
        )
        if len(self.grouped_units) > 0:
            self.logger.info(f"Replacing existing grouping with {group_index}")
        self.grouped_units = group_index
