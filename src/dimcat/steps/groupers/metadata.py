from numbers import Number
from typing import Dict, List, Sequence

import pandas as pd
from dimcat import Dataset
from dimcat.data.resources import DimcatIndex
from dimcat.steps.groupers import CustomPieceGrouper
from dimcat.utils import get_middle_composition_year
from typing_extensions import Self


class YearGrouper(CustomPieceGrouper):
    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[Number, List[tuple]],
        level_names: Sequence[str] = ("middle_composition_year", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a YearGrouper from a dictionary of piece groups.

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
        level_name: str = "middle_composition_year",
        grouped_units: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, grouped_units=grouped_units, **kwargs)

    def fit_to_dataset(self, dataset: Dataset) -> None:
        metadata = dataset.get_metadata(raw=True)
        sorted_composition_years = get_middle_composition_year(metadata).sort_values()
        grouping = sorted_composition_years.groupby(
            sorted_composition_years, sort=True
        ).groups
        group_index = DimcatIndex.from_grouping(
            grouping, ("middle_composition_year", "corpus", "piece")
        )
        if len(self.grouped_units) > 0:
            self.logger.info(f"Replacing existing grouping with {group_index}")
        self.grouped_units = group_index
