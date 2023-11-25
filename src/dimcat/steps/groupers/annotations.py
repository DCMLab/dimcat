import logging
from typing import Dict, Hashable, List, Sequence

import pandas as pd
from dimcat.data.resources import DimcatIndex
from dimcat.data.resources.base import D, FeatureName
from dimcat.data.resources.dc import UnitOfAnalysis
from dimcat.steps.groupers.base import CriterionGrouper
from typing_extensions import Self

logger = logging.getLogger(__name__)


class HasCadenceAnnotations(CriterionGrouper):
    """Allows for grouping by specifying a {group_name: [piece_index_tuples]} dictionary."""

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
