import logging
from typing import Dict, List, Sequence

import pandas as pd
from dimcat.dataset.processed import GroupedDataset
from dimcat.pipeline import PipelineStep
from dimcat.resources.base import DimcatIndex, IndexField
from dimcat.resources.features import Feature
from dimcat.resources.utils import make_index_from_grouping_dict
from dimcat.utils import check_name
from marshmallow import fields
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Grouper(PipelineStep):
    new_dataset_type = GroupedDataset
    new_resource_type = None
    output_package_name = None

    class Schema(PipelineStep.Schema):
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

    def dispatch(self, resource: Feature) -> Feature:
        """Apply the grouper to a Feature."""
        result_constructor = self.get_new_resource_type(resource)
        results = self.apply_grouper(resource)
        result_name = self.resource_name_factory(resource)
        return result_constructor.from_dataframe(
            df=results,
            resource_name=result_name,
        )


class CustomPieceGrouper(Grouper):
    class Schema(Grouper.Schema):
        grouped_pieces = IndexField()

    @classmethod
    def from_dict(
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
        sort: By default the returned MultiIndex is sorted. Set False to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouping = make_index_from_grouping_dict(
            grouping=piece_groups,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=level_names[0], grouped_pieces=DimcatIndex(grouping))

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
